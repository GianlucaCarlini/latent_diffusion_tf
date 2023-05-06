import tensorflow as tf
from tensorflow.keras.models import Model
from .blocks import ResidualBlock, SinusoidalEmbedding


def get_unet(input_shape, depths, initial_dim, output_classes=8, **kwargs):

    embed_dims = [initial_dim * 2**i for i in range(len(depths))]

    image = tf.keras.Input(shape=input_shape, name="Input")
    noise = tf.keras.Input(shape=(1, 1, 1), dtype=tf.float32)

    y = SinusoidalEmbedding(embed_dim=embed_dims[0])(noise)
    y = tf.keras.layers.UpSampling2D(size=(input_shape[0], input_shape[1]))(y)

    x = tf.keras.layers.Conv2D(filters=embed_dims[0], kernel_size=1)(image)
    x = tf.keras.layers.Concatenate()([x, y])

    skips = []

    for i, depth in enumerate(depths):

        for _ in range(depth):

            x = ResidualBlock(output_dim=embed_dims[i], activation="gelu")(x)

        if i < len(depths) - 1:
            skips.append(x)

            x = tf.keras.layers.Conv2D(
                filters=embed_dims[i], kernel_size=2, strides=2, padding="same"
            )(x)

    depths.reverse()
    embed_dims.reverse()

    for i, depth in enumerate(depths[1:]):

        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Concatenate()([x, skips.pop()])

        for _ in range(depth):

            x = ResidualBlock(output_dim=embed_dims[i + 1], activation="gelu")(x)

    x = tf.keras.layers.Conv2D(filters=output_classes, kernel_size=1)(x)

    unet = tf.keras.models.Model(inputs=[image, noise], outputs=[x])

    return unet
