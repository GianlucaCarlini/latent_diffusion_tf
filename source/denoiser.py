import tensorflow as tf
from tensorflow.keras.models import Model
from .blocks import ResidualBlock, SinusoidalEmbedding, AttentionBlock
from .blocks import SwinBasicLayer


def get_unet(input_shape, depths, initial_dim, output_classes=8, **kwargs):

    embed_dims = [initial_dim * 2**i for i in range(len(depths))]

    image = tf.keras.Input(shape=input_shape, name="Input")
    noise = tf.keras.Input(shape=(1, 1, 1), dtype=tf.float32)

    y = SinusoidalEmbedding(embed_dim=embed_dims[0] // 2)(noise)
    y = tf.keras.layers.UpSampling2D(
        size=(input_shape[0], input_shape[1]), interpolation="nearest"
    )(y)

    x = tf.keras.layers.Conv2D(filters=embed_dims[0] // 2, kernel_size=1)(image)
    x = tf.keras.layers.Concatenate()([x, y])

    skips = []
    times = []

    for i, depth in enumerate(depths):

        if i > 0:
            x = tf.keras.layers.Concatenate()([x, y])

        for _ in range(depth):

            x = ResidualBlock(output_dim=embed_dims[i], activation="swish")(x)

        if i > 0:
            x = AttentionBlock(output_dim=embed_dims[i])(x)

        if i < len(depths) - 1:
            skips.append(x)
            times.append(y)

            y = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(y)

            x = tf.keras.layers.Conv2D(
                filters=embed_dims[i], kernel_size=2, strides=2, padding="same"
            )(x)

    depths.reverse()
    embed_dims.reverse()

    for i, depth in enumerate(depths[1:]):

        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bicubic")(x)
        x = tf.keras.layers.Concatenate()([x, skips.pop()])
        x = tf.keras.layers.Concatenate()([x, times.pop()])

        for _ in range(depth):

            x = ResidualBlock(output_dim=embed_dims[i + 1], activation="swish")(x)

    x = tf.keras.layers.Conv2D(filters=output_classes, kernel_size=1)(x)

    unet = tf.keras.models.Model(inputs=[image, noise], outputs=[x])

    return unet


def get_swin_unet(input_shape, depths, initial_dim, output_classes=8, **kwargs):

    embed_dims = [initial_dim * 2**i for i in range(len(depths))]

    image = tf.keras.Input(shape=input_shape, name="Input")
    noise = tf.keras.Input(shape=(1, 1, 1), dtype=tf.float32)

    y = SinusoidalEmbedding(embed_dim=embed_dims[0] // 2)(noise)
    y = tf.keras.layers.UpSampling2D(
        size=(input_shape[0], input_shape[1]), interpolation="nearest"
    )(y)

    x = tf.keras.layers.Conv2D(filters=embed_dims[0] // 2, kernel_size=1)(image)
    x = tf.keras.layers.Concatenate()([x, y])

    skips = []
    times = []

    for i, depth in enumerate(depths):

        if i > 0:
            x = tf.keras.layers.Concatenate()([x, y])

        _, H, W, C = x.shape

        x = tf.reshape(x, shape=[-1, H * W, C])

        x = SwinBasicLayer(
            x,
            dim=embed_dims[i],
            input_resolution=[H, W],
            depth=depth,
            num_heads=embed_dims[i] // 24,
            window_size=8,
            name=f'encoder_block_{i}'
        )

        C = x.shape[-1]

        if i < len(depths) - 1:

            x = tf.reshape(x, shape=[-1, H, W, C])

            skips.append(x)
            times.append(y)

            y = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(y)

            x = tf.keras.layers.Conv2D(
                filters=embed_dims[i], kernel_size=2, strides=2, padding="same"
            )(x)

    depths.reverse()
    embed_dims.reverse()

    for i, depth in enumerate(depths[1:]):

        C = x.shape[-1]

        x = tf.reshape(x, shape=[-1, H, W, C])
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bicubic")(x)
        x = tf.keras.layers.Concatenate()([x, skips.pop()])
        x = tf.keras.layers.Concatenate()([x, times.pop()])

        _, H, W, C = x.shape

        x = tf.reshape(x, shape=[-1, H * W, C])

        x = SwinBasicLayer(
            x,
            dim=embed_dims[i + 1],
            input_resolution=[H, W],
            depth=depth,
            num_heads=embed_dims[i + 1] // 24,
            window_size=8,
            name=f'decoder_block_{i}'
        )

        C = x.shape[-1]

        x = tf.reshape(x, shape=[-1, H, W, C])

    x = tf.keras.layers.Conv2D(filters=output_classes, kernel_size=1)(x)

    unet = tf.keras.models.Model(inputs=[image, noise], outputs=[x])

    return unet
