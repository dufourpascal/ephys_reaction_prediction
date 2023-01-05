import glob
import re
import numpy as np
import csv
# import matplotlib.pyplot as plt
import os
import random
from cwt import cmw_convolution_trials
from typing import Optional
from tensorflow import keras


def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    x = x - mean
    x = x / std
    return x


class DataGenerator(keras.utils.Sequence):
    'Generates data for Training and Inference'
    def __init__(
        self,
        x, y,
        batch_size=32,
        shuffle=False,
        data_augmentation=True,
        out_shape=(42, 42, 2),
        scale_min=0.25,
        scale_max=4.0,
    ):
        self.data_augmentation = data_augmentation
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shuffle = shuffle

        self.y1 = np.squeeze(y == 1.0)
        self.y0 = np.invert(self.y1)

        self.x0 = x[self.y0, :, :, :]
        self.x1 = x[self.y1, :, :, :]
        if self.shuffle:
            np.random.shuffle(self.x0)
            np.random.shuffle(self.x1)

        self.out_shape = out_shape
        self.n_class0 = np.sum(self.y0)
        self.n_class1 = np.sum(self.y1)
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        n_samples_class = max(self.x0.shape[0], self.x1.shape[0])
        # each sample will be shown at once or more during each epoch
        total_samples = n_samples_class * 2
        return int(np.floor(total_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X, y = self._data_generation(index)
        return X, y

    def _data_generation(self, index):
        'Generates data containing batch_size samples with equal probability'
        # Initialization
        n_classes = 2
        offset = 5
        assert self.batch_size % n_classes == 0
        height, width, n_channels = self.out_shape

        x = np.empty((self.batch_size, height, width, n_channels),
                     dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        batch_size_data = int(self.batch_size / n_classes)

        idx_data_start = int(batch_size_data * index)
        idx_data_end = int(batch_size_data * (index + 1))

        idx_data_start_0 = idx_data_start % self.x0.shape[0]
        idx_data_end_0 = idx_data_end % self.x0.shape[0]
        if idx_data_start_0 > idx_data_end_0:  # in case only end idx wraps
            idx_data_start_0 = 0
            idx_data_end_0 = batch_size_data

        idx_data_start_1 = idx_data_start % self.x1.shape[0]
        idx_data_end_1 = idx_data_end % self.x1.shape[0]
        if idx_data_start_1 > idx_data_end_1:
            idx_data_start_1 = 0
            idx_data_end_1 = batch_size_data

        # assert idx_data_end < self.x0.shape[0]
        # assert idx_data_end < self.x1.shape[0]

        if self.data_augmentation:
            elements = range(offset, self.x0.shape[2] - width - offset)
            start_w = random.choice(elements)
        else:
            start_w = offset

        x0 = self.x0[idx_data_start_0:idx_data_end_0, :,
                     start_w:start_w + width, :]
        x1 = self.x1[idx_data_start_1:idx_data_end_1, :,
                     start_w:start_w + width, :]

        y = np.zeros((self.batch_size), dtype=int)
        y[:batch_size_data] = 0
        y[batch_size_data:] = 1

        x = np.concatenate([x0, x1], axis=0)

        for i in range(x.shape[0]):
            # adjust mean to 0
            # x[i, :, :, 0] = x[i, :, :, 0] - np.mean(x[i, :, :, 0])
            # x[i, :, :, 1] = x[i, :, :, 1] - np.mean(x[i, :, :, 1])

            x[i, :, :, 0] = x[i, :, :, 0] / np.std(x[i, :, :, 0])
            x[i, :, :, 1] = x[i, :, :, 1] / np.std(x[i, :, :, 1])

        if self.data_augmentation:
            for i in range(x.shape[0]):
                if random.randint(0, 1):
                    # scale down
                    scale = random.uniform(self.scale_min, 1.0)
                else:
                    # scale up
                    scale = random.uniform(1.0, self.scale_max)

                x[i, :, :, 0] = x[i, :, :, 0] * scale
                x[i, :, :, 1] = x[i, :, :, 1] * scale

        return x, y


class Dataset:
    def __init__(
        self,
        data_base_dir: str,
        raw_pretrain_dir: str,
        from_cache: bool = False,
        cache_dir: str = 'spectral',
        limit: Optional[int] = None,
        log10: bool = True,
        band=None,
        band_width=0,
    ):
        self.data_base_dir = data_base_dir
        self.raw_pretrain_dir = raw_pretrain_dir
        self.cache_dir = cache_dir
        self.from_cache = from_cache
        save_dir = os.path.join(data_base_dir, cache_dir)
        if from_cache:
            self.data, self.groups, self.classes = \
                Dataset._load_cache_spectral(save_dir)

        else:
            raw_dir = os.path.join(data_base_dir, raw_pretrain_dir)

            if self.raw_pretrain_dir == 'sleep_wake':
                data_raw, self.groups, self.classes = \
                    self._load_raw_sleep_wake(raw_dir, limit=limit)

            elif self.raw_pretrain_dir == 'reaction':
                data_raw, self.groups, self.classes = \
                    self._load_raw_reaction(raw_dir, limit=limit)

            self.data = self._create_spectral_data(data_raw)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            Dataset._save_cache_spectral(
                save_dir, self.data, self.groups, self.classes)

        if log10:
            self.data[:, :, :, 0] = np.log10(self.data[:, :, :, 0])
            self.data[:, :, :, 2] = np.log10(self.data[:, :, :, 2])

        # remove part of the data
        if band_width > 0:
            band_max = min(band+band_width, self.data.shape[1])
            self.data[:, band:band_max, :, :] = 0.0

        print(f'Loaded {self.data.shape[0]} samples, '
              f'shape: {self.data.shape}')

    def describe(self):
        print('==========================================')
        if self.from_cache:
            full_cache_dir = os.path.join(self.data_base_dir, self.cache_dir)
            print(f'Dataset loaded from cache {full_cache_dir}')
        else:
            full_data_dir = os.path.join(
                self.data_base_dir, self.raw_pretrain_dir)
            print(f'Dataset loaded from raw data {full_data_dir}')

        n_samples, height, width, channels = self.data.shape
        print(f'Total of {n_samples} of size {height}x{width}')

        if isinstance(self.classes, dict):
            n_reaction = np.sum(self.classes['reaction'])
            n_no_reaction = np.sum(np.invert(self.classes['reaction']))
            print(f'Classes: {n_reaction} samples reaction,'
                  f'{n_no_reaction} samples no-reaction')
        else:
            n_sleep = np.sum(self.classes == 'n')
            n_wake = np.sum(self.classes == 'w')
            print(f'Classes: {n_sleep} samples wake, {n_wake} samples sleep')

        print('==========================================')

    @staticmethod
    def _load_cache_spectral(save_dir: str):
        data = np.load(os.path.join(save_dir, 'data.npy'))
        data = data.astype(np.float32)
        groups = np.load(os.path.join(save_dir, 'groups.npy'))
        groups = np.squeeze(groups)
        classes = np.load(os.path.join(save_dir, 'classes.npy'), allow_pickle=True)
        if classes.ndim == 0:
            classes = classes.tolist()
        return data, groups, classes

    @staticmethod
    def _save_cache_spectral(save_dir: str, data, groups, classes):
        np.save(os.path.join(save_dir, 'data.npy'), data)
        np.save(os.path.join(save_dir, 'groups.npy'), groups)
        np.save(os.path.join(save_dir, 'classes.npy'), classes)
        return data, groups, classes

    def _create_spectral_data(self, data: np.array):
        print('CWT for ACC data...')
        freqs_acc = cmw_convolution_trials(data[:, :, 0].T)
        print('CWT for S1HL data...')
        freqs_s1hl = cmw_convolution_trials(data[:, :, 1].T)
        data_freq = np.concatenate([freqs_acc, freqs_s1hl], axis=3)
        data_freq = np.moveaxis(data_freq, 2, 0)
        return data_freq

    @staticmethod
    def list_files(data_dir: str):
        npy_files = glob.glob(os.path.join(data_dir, '*.npy'))
        return npy_files

    def __aggregate_stim_idx(
        self,
        rows,
        lateralization_ACC='contralateral',
        lateralization_S1HL='contralateral',
    ):
        def get_key(rows, brain_area, lateralization):
            for row in rows:
                if row['brain_area'] == brain_area and \
                   row['lateralization'] == lateralization:
                    idx = row['']
                    if idx:
                        return int(idx)
                    else:
                        return None

        idx_acc = get_key(rows, 'ACC', lateralization_ACC)
        idx_s1hl = get_key(rows, 'S1HL', lateralization_S1HL)

        return idx_acc, idx_s1hl

    def _load_raw_reaction(self, data_dir: str, limit: Optional[int] = None):
        npy_files = Dataset.list_files(data_dir)
        regex_str = r'.*_(\d*)_.*.npy'

        data_list = []
        group_list = []
        labels_pre_list = []
        labels_post_list = []
        labels_reaction_list = []
        labels_stim_type_list = []

        for f in npy_files:
            f_csv = f.replace('.npy', '.csv')
            match = re.match(regex_str, f)
            animal_id = match.groups()[0]
            animal_id = int(animal_id)

            print(f'Loading {f}...')
            data_exp = np.load(f)
            metadata_by_stim_idx = {}

            with open(f_csv, 'r') as fid:
                reader = csv.DictReader(fid, delimiter=',')
                for row in reader:
                    # per stim_idx, get a list of rows
                    stim_idx = row['stim_idx']
                    if stim_idx not in metadata_by_stim_idx:
                        metadata_by_stim_idx[stim_idx] = []
                    metadata_by_stim_idx[stim_idx].append(row)

            data = []
            labels = []
            for stim_idx, rows in metadata_by_stim_idx.items():
                label = {}
                label['stim_type'] = rows[0]['stim_type']
                category = rows[0]['category']
                state_pre, state_post, reaction = category
                label['state_pre'] = state_pre
                label['state_post'] = state_post
                label['reaction'] = reaction
                idx_acc, idx_s1hl = self.__aggregate_stim_idx(
                    rows,
                    lateralization_ACC='contralateral',
                    lateralization_S1HL='contralateral',
                )

                if isinstance(idx_acc, int) and \
                   isinstance(idx_s1hl, int):
                    data_acc_exp = data_exp[:, idx_acc]
                    data_s1hl_exp = data_exp[:, idx_s1hl]
                    data_acc_s1hl = np.stack((data_acc_exp, data_s1hl_exp), axis=1)
                    if np.isnan(data_acc_s1hl).any():
                        continue
                    data.append(data_acc_s1hl)
                    labels.append(label)

            if not data:
                continue
            data = np.stack(data, axis=0)

            n_samples = data.shape[0]

            labels_pre = np.array([x['state_pre'] for x in labels])
            labels_pre_list.append(labels_pre)

            labels_post = np.array([x['state_post'] for x in labels])
            labels_post_list.append(labels_post)

            labels_reaction = np.array([x['reaction'] == 'F' for x in labels])
            labels_reaction_list.append(labels_reaction)

            labels_stim_type = np.array([x['stim_type'] for x in labels])
            labels_stim_type_list.append(labels_stim_type)

            data_list.append(data)

            group_exp = np.full((n_samples, 1), animal_id)
            group_list.append(group_exp)

        data_raw = np.concatenate(data_list, axis=0)

        groups_raw = np.concatenate(group_list, axis=0)
        labels_pre_raw = np.concatenate(labels_pre_list, axis=0)
        labels_post_raw = np.concatenate(labels_post_list, axis=0)
        labels_reaction_raw = np.concatenate(labels_reaction_list, axis=0)
        labels_stim_type_raw = np.concatenate(labels_stim_type_list, axis=0)

        if limit:
            data_raw = data_raw[:limit, :]
            groups_raw = groups_raw[:limit, :]
            labels_pre_raw = labels_pre_raw[:limit]
            labels_post_raw = labels_post_raw[:limit]
            labels_reaction_raw = labels_reaction_raw[:limit]
            labels_stim_type_raw = labels_stim_type_raw[:limit]

        labels_raw = {
            'pre': labels_pre_raw,
            'post': labels_post_raw,
            'reaction': labels_reaction_raw,
            'stim_type': labels_stim_type_raw
        }

        return data_raw, groups_raw, labels_raw

    def _load_raw_sleep_wake(self, data_dir: str, limit: Optional[int] = None):
        npy_files = Dataset.list_files(data_dir)
        regex_str = r'.*_(\d*)_\d*_([w|n]).npy'

        data_list = []
        group_list = []
        class_list = []

        for f in npy_files:
            match = re.match(regex_str, f)
            animal_id, sleep_state = match.groups()
            animal_id = int(animal_id)

            print(f'Loading {f}...')
            data_exp = np.load(f)
            n_samples = data_exp.shape[0]
            group_exp = np.full((n_samples, 1), animal_id)
            class_exp = np.full((n_samples, 1), sleep_state)
            data_list.append(data_exp)
            group_list.append(group_exp)
            class_list.append(class_exp)

        data_raw = np.concatenate(data_list, axis=0)
        groups_raw = np.concatenate(group_list, axis=0)
        classes_raw = np.concatenate(class_list, axis=0)
        if limit:
            data_raw = data_raw[:limit, :]
            groups_raw = groups_raw[:limit, :]
            classes_raw = classes_raw[:limit, :]

        print(f'Loaded {len(npy_files)} files: {data_raw.shape[0]} samples.')
        return data_raw, groups_raw, classes_raw

    def train_test_gen(
        self,
        groups_train,
        batch_size=32,
        shuffle=False,
        data_augmentation=True,
        width=42,
        offset_x=5,
        channels=(0, 2),
        scale_min=0.25,
        scale_max=4.0,
        filters={},
    ):
        x, y, xt, yt, gt = self.train_test_set(groups_train, filters=filters)

        height = x.shape[1]

        train_datagen = DataGenerator(
            x, y,
            out_shape=(height, width, len(channels)),
            batch_size=batch_size,
            shuffle=shuffle,
            data_augmentation=data_augmentation,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        idx_end_x = xt.shape[2] - offset_x
        idx_start_x = idx_end_x - width
        xt = xt[:, :, idx_start_x:idx_end_x, :]
        for i in range(xt.shape[0]):
            # xt[i, :, :, 0] = xt[i, :, :, 0] - np.mean(xt[i, :, :, 0])
            # xt[i, :, :, 1] = xt[i, :, :, 1] - np.mean(xt[i, :, :, 1])

            xt[i, :, :, 0] = xt[i, :, :, 0] / np.std(xt[i, :, :, 0])
            xt[i, :, :, 1] = xt[i, :, :, 1] / np.std(xt[i, :, :, 1])

        yt = np.squeeze(yt) * 1
        return train_datagen, xt, yt, gt

    def train_test_set(
        self,
        groups_train,
        channels=(0, 2),
        filters={},
    ):
        train_ind = np.squeeze(np.isin(self.groups, groups_train))
        test_ind = np.invert(train_ind)

        data = self.data[:, :, :, channels]

        if isinstance(self.classes, dict):
            idx_filtered = np.ones(self.classes['reaction'].shape, dtype=bool)
            for k, v in filters.items():
                idx_filtered = np.logical_and(idx_filtered,
                                              self.classes[k] == v)
            labels = self.classes['reaction']
            train_ind = np.logical_and(train_ind, idx_filtered)
            test_ind = np.logical_and(test_ind, idx_filtered)
        else:
            labels = self.classes == 'w'

        labels = labels * 1

        X_train = data[train_ind, :, :, :]
        y_train = labels[train_ind]

        X_test = data[test_ind, :, :, :]
        y_test = labels[test_ind]
        groups_test = self.groups[test_ind]

        return X_train, y_train, X_test, y_test, groups_test


if __name__ == '__main__':
    pass
    # dataset = Dataset(
    #     '/media/pascal/T7 Shield/datasets/ephys',
    #     'reaction',
    #     cache_dir='spectral_reaction',
    #     from_cache=True,
    #     limit=None,
    # )
    # dataset.describe()

    # groups = np.unique(dataset.groups)
    # train_groups = groups[:10]
    # print(train_groups)

    # train_gen, xt, yt, gt = dataset.train_test_gen(
    #     train_groups,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     data_augmentation=True,
    #     width=42,
    #     offset_x=5,
    #     channels=(0, 2),
    # )

    # batch_x, batch_y = train_gen[10]
    # print(batch_x, batch_y)

    # import matplotlib.pyplot as plt

    # for i in range(4):
    #     fig = plt.figure()
    #     pos = plt.imshow(batch_x[i, :, :, 0], origin='lower', aspect='auto')
    #     fig.colorbar(pos)
    #     fig = plt.figure()
    #     pos = plt.imshow(batch_x[i, :, :, 1], origin='lower', aspect='auto')
    #     fig.colorbar(pos)
    # plt.show()
    # print()
