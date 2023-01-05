from experiment import run_experiment

import numpy as np
import json
import os


if __name__ == '__main__':
    all_groups = np.array([1, 2, 3, 4, 5, 6, 7, 22, 26, 27, 34, 35, 37, 38,
                           39, 40])

    epochs = 8
    batch_size = 32
    mlflow_log = False
    log_scale = False
    batch_norm = False
    n_filters = (4, 8, 8, 16)
    n_dense = (128, 64)
    optimizer = 'adam'
    filters = {'pre': 'n'}
    band_width = 0
    data_augmentation_args = {
        'enable': True,
        'noise_stddev': 0.25,
        'scale_min': 0.5,
        'scale_max': 2.0,
    }

    # cross validation
    n_folds = 8
    groups_per_folds = len(all_groups) / n_folds
    print(f'all groups: {all_groups}')

    results = {}

    run_configs = []
    band_width = 8
    step_size = 2
    if band_width:
        band_bottoms = range(0, 42, step_size)
    else:
        band_bottoms = [None]

    for band_bottom in band_bottoms:
        print(f'=== band {band_bottom}, width {band_width} ===')
        group_start_idx = 0.0
        for i in range(n_folds):
            group_end_idx = group_start_idx + groups_per_folds
            test_groups = all_groups[int(group_start_idx):int(group_end_idx)]
            train_ind = np.invert(np.isin(all_groups, test_groups))
            train_groups = all_groups[np.squeeze(train_ind)]
            print(f'==Train: {train_groups}, Test: {test_groups}==')
            group_start_idx = group_end_idx

            result = run_experiment(
                train_groups=train_groups,
                band=band_bottom,
                band_with=band_width,
                mlflow_log=mlflow_log,
                log_scale=log_scale,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                batch_norm=batch_norm,
                optimizer=optimizer,
                n_filters=n_filters,
                n_dense=n_dense,
                data_augmentation_args=data_augmentation_args,
                filters=filters,
            )
            results = {**results, **result}

        results_dir = '/media/pascal/T7 Shield/datasets/ephys/results'
        os.makedirs(results_dir, exist_ok=True)
        result_filename = f'result_band_{band_bottom}_{band_width}'
        for k, v in filters.items():
            result_filename += f'_{k}_{v}'
        result_filename += '.json'
        result_filename = os.path.join(results_dir, result_filename)
        with open(result_filename, 'w') as fid:
            json.dump(results, fid)
