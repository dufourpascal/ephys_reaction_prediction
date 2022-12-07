from tensorflow import keras


def _conv_block(
    input_layer,
    n_filters,
    pool=True,
    noise_stddev=None,
    padding='same'
):
    x = keras.layers.Conv2D(
        filters=n_filters, kernel_size=5, padding=padding
    )(input_layer)
    # x = keras.layers.BatchNormalization()(x)

    if noise_stddev:
        x = keras.layers.GaussianNoise(stddev=noise_stddev)(x)

    x = keras.layers.ReLU()(x)
    if pool:
        x = keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))(x)
    return x


def make_model(
    input_shape,
    batch_norm=False,
    noise_stddev=None,
    n_filters=(4, 4, 8, 16),
    n_dense=(32,),
):
    input = keras.layers.Input(input_shape)

    if batch_norm:
        x = keras.layers.BatchNormalization()(input)
    else:
        x = input

    for filters in n_filters[0:-1]:
        x = _conv_block(x, n_filters=filters, noise_stddev=noise_stddev)

    # x = _conv_block(x, n_filters=4, noise_stddev=noise_stddev)
    # x = _conv_block(x, n_filters=4, noise_stddev=noise_stddev)
    # x = _conv_block(x, n_filters=8, noise_stddev=noise_stddev)

    x = _conv_block(x, n_filters=n_filters[-1], pool=False, padding='valid')

    # gap = keras.layers.GlobalAveragePooling2D()(conv4)
    x = keras.layers.Flatten()(x)
    for dense in n_dense:
        x = keras.layers.Dense(dense)(x)

    out = keras.layers.Dense(2, activation='softmax')(x)

    return keras.models.Model(inputs=input, outputs=out)


if __name__ == '__main__':
    m = make_model(input_shape=(42, 42, 2))
    m.summary()
