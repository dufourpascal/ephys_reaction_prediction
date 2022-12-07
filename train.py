from experiment import run_experiment

if __name__ == '__main__':
    epochs = 16
    batch_size = 32
    mlflow_log = True
    log_scale = True
    batch_norm = False
    n_filters = (4, 8, 8, 16)
    n_dense = (128, 64)
    optimizer = 'adam'
    data_augmentation_args = {
        'enable': True,
        'noise_stddev': 0.25,
        'scale_min': 0.5,
        'scale_max': 2.0,
    }

    run_experiment(
        model_name='pretraining_generic',
        mlflow_log=mlflow_log,
        log_scale=log_scale,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        batch_norm=batch_norm,
        optimizer=optimizer,
        n_filters=n_filters,
        n_dense=n_dense,
        data_augmentation_args=data_augmentation_args
    )
