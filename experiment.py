from model import make_model
from dataset import Dataset
import numpy as np
from mlflow.tensorflow import autolog
from mlflow import log_params, start_run, end_run
import os.path
from tensorflow import keras


def predict_on_groups(m, xt, yt, gt):
    y_pred = m.predict(xt)
    y_pred = (y_pred[:, 0] < 0.5) * 1.0
    y_correct = y_pred == yt

    result = {}
    for group in np.unique(gt):
        idx_group = gt == group
        n_samples = np.sum(idx_group)
        n_correct = np.sum(y_correct[idx_group])
        accuracy = float(n_correct / n_samples)
        result[int(group)] = accuracy
        print(f'Group {group}: Accuracy {n_correct / n_samples}')

    n_samples = len(y_pred)
    n_correct = np.sum(y_correct)
    print(f'Overall accuracy {n_correct / n_samples}')

    return result


def k_fold_experiment(
    log_scale=True,
    k=8
):
    dataset = Dataset(
        '/media/pascal/T7 Shield/datasets/ephys',
        'sleep_wake',
        from_cache=True,
        limit=None,
        log10=log_scale,
    )
    groups = np.unique(dataset.groups)
    groups_per_k = len(groups) / k
    group_start_idx = 0.0
    print(f'all groups: {groups}')
    for i in range(k):
        group_end_idx = group_start_idx + groups_per_k
        test_groups = groups[int(group_start_idx):int(group_end_idx)]
        train_ind = np.invert(np.isin(groups, test_groups))
        train_groups = groups[np.squeeze(train_ind)]
        print(train_groups, test_groups)
        group_start_idx = group_end_idx

    # train_groups = groups[:n_train]
    # input_width = 42
    # print(train_groups)


def get_pretrained_model(
    train_groups,
    model_dir='/media/pascal/T7 Shield/datasets/ephys/models',
    use_cache=True,
    log_scale=True,
    epochs=16,
    batch_size=32,
    verbose=1,
    batch_norm=False,
    optimizer='adam',
    n_filters=(4, 4, 8, 16),
    n_dense=(32,),
    data_augmentation_args={'enable': False},
    mlflow_log=True,
):
    train_groups.sort()
    model_name = 'wake_sleep_'
    model_name += '_'.join([str(x) for x in train_groups])

    model_filename = os.path.join(model_dir, model_name)

    # try to load the cached model
    if use_cache and os.path.exists(model_filename):
        print(f'Loading pretrained model {model_filename}')
        m = keras.models.load_model(model_filename)
        return m

    # if we don't have the model saved, train it from scratch
    print(f'Recomputing pretrained model {model_filename}')

    dataset = Dataset(
        '/media/pascal/T7 Shield/datasets/ephys',
        'sleep_wake',
        from_cache=True,
        limit=None,
        log10=log_scale,
    )

    input_width = 42
    print(train_groups)
    train_gen, xt, yt, gt = dataset.train_test_gen(
        train_groups,
        batch_size=batch_size,
        shuffle=True,
        data_augmentation=data_augmentation_args['enable'],
        width=input_width,
        offset_x=5,
        channels=(0, 2),
        scale_min=data_augmentation_args['scale_min'],
        scale_max=data_augmentation_args['scale_max'],
    )

    m = make_model(
        input_shape=(42, input_width, 2),
        batch_norm=batch_norm,
        n_filters=n_filters,
        n_dense=n_dense,
        noise_stddev=data_augmentation_args['noise_stddev'],
    )
    m.summary()

    m.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    if mlflow_log:
        start_run()
        autolog(registered_model_name=model_name)
        log_params({
            **data_augmentation_args,
            'batch_norm': batch_norm,
            'n_filters': n_filters,
            'n_dense': n_dense,
        })

    m.fit(
        train_gen,
        validation_data=(xt, yt),
        epochs=epochs,
        batch_size=batch_size,
        use_multiprocessing=True,
        workers=6,
        verbose=verbose,
    )

    print(f'Saving pretrained model {model_filename}')
    m.save(model_filename)

    predict_on_groups(m, xt, yt, gt)
    if mlflow_log:
        end_run()

    return m


def run_experiment(
    train_groups,
    band=None,
    band_with=0,
    mlflow_log=True,
    log_scale=True,
    epochs=16,
    batch_size=32,
    verbose=1,
    batch_norm=False,
    optimizer='adam',
    n_filters=(4, 4, 8, 16),
    n_dense=(32,),
    data_augmentation_args={'enable': False},
    filters={},
):
    data_augmentation_args_pretrain = {
        'enable': True,
        'noise_stddev': 0.25,
        'scale_min': 0.5,
        'scale_max': 2.0,
    }

    m = get_pretrained_model(
        train_groups,
        model_dir='/media/pascal/T7 Shield/datasets/ephys/models',
        use_cache=True,
        log_scale=log_scale,
        epochs=16,
        batch_size=batch_size,
        verbose=1,
        batch_norm=False,
        optimizer='adam',
        n_filters=n_filters,
        n_dense=n_dense,
        data_augmentation_args=data_augmentation_args_pretrain,
        mlflow_log=mlflow_log,
    )

    # finetune on reaction dataset
    dataset = Dataset(
            '/media/pascal/T7 Shield/datasets/ephys',
            'reaction',
            cache_dir='spectral_reaction',
            from_cache=True,
            limit=None,
            log10=log_scale,
            band=band,
            band_width=band_with,
        )

    train_gen, xt, yt, gt = dataset.train_test_gen(
        train_groups,
        batch_size=batch_size,
        shuffle=True,
        data_augmentation=data_augmentation_args['enable'],
        width=42,
        offset_x=5,
        channels=(0, 2),
        scale_min=data_augmentation_args['scale_min'],
        scale_max=data_augmentation_args['scale_max'],
        filters=filters,
    )

    if not len(yt):
        # no test data (might depend on recordings and filtering)
        print(f'!!!NO TEST DATA for training group {train_groups}, '
              'will not run training!!!')
        return

    print('Running reaction prediction training...')

    train_groups.sort()
    if mlflow_log:
        model_name = 'reaction_' + '_'.join([str(x) for x in train_groups])
        start_run()
        autolog(registered_model_name=model_name)
        log_params({
            **data_augmentation_args,
            'batch_norm': batch_norm,
            'n_filters': n_filters,
            'n_dense': n_dense,
            **filters,
        })

    m.fit(
        train_gen,
        validation_data=(xt, yt),
        epochs=epochs,
        batch_size=batch_size,
        use_multiprocessing=True,
        workers=6,
        verbose=verbose,
    )

    results = predict_on_groups(m, xt, yt, gt)
    if mlflow_log:
        end_run()

    return results


if __name__ == '__main__':
    k_fold_experiment(k=8)
