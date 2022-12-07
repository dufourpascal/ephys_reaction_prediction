from model import make_model
from dataset import Dataset
import numpy as np
from mlflow.keras import autolog
from mlflow import log_params, start_run, end_run


def predict_on_groups(m, xt, yt, gt):
    y_pred = m.predict(xt)
    y_pred = (y_pred[:, 0] < 0.5) * 1.0
    y_correct = y_pred == yt

    for group in np.unique(gt):
        idx_group = gt == group
        n_samples = np.sum(idx_group)
        n_correct = np.sum(y_correct[idx_group])
        print(f'Group {group}: Accuracy {n_correct / n_samples}')

    n_samples = len(y_pred)
    n_correct = np.sum(y_correct)
    print(f'Overall accuracy {n_correct / n_samples}')


def run_experiment(
    model_name,
    mlflow_log=True,
    log_scale=True,
    epochs=16,
    batch_size=32,
    verbose=1,
    batch_norm=False,
    optimizer='adam',
    n_filters=(4, 4, 8, 16),
    n_dense=(32,),
    data_augmentation_args={'enabled': False},
):
    dataset = Dataset(
        '/media/pascal/T7 Shield/datasets/ephys',
        'sleep_wake',
        from_cache=True,
        limit=None,
        log10=log_scale,
    )
    groups = np.unique(dataset.groups)
    n_train = 10
    # train_groups = groups[-n_train:]
    train_groups = groups[:n_train]
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
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
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

    predict_on_groups(m, xt, yt, gt)
    if mlflow_log:
        end_run()
