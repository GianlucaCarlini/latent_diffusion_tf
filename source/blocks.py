import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class ResidualBlock(layers.Layer):
    def __init__(self, output_dim, activation="relu", **kwargs) -> None:
        """Standard residual block with pre-activation https://paperswithcode.com/method/residual-block

        Args:
            output_dim (int): The output channel dimension.
            norm_layer (tf.keras.layers.Layer, optional): The normalization layer
                to be applied befor the activation and the convolution. Defaults to None.
                If None, LayerNormalization is applied.
            activation (str, optional): The activation function for the residual.
                Defaults to "swish".
        """
        super().__init__(**kwargs)

        self.outpud_dim = output_dim

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.act1 = tf.keras.activations.get(activation)
        self.conv1 = layers.Conv2D(output_dim, kernel_size=3, padding="same")
        self.act2 = layers.Activation(activation=activation)
        self.conv2 = layers.Conv2D(output_dim, kernel_size=3, padding="same")
        self.proj = layers.Conv2D(output_dim, kernel_size=1)

    def call(self, inputs):

        x = self.norm1(inputs)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        if self.outpud_dim != inputs.shape[-1]:
            inputs = self.proj(inputs)

        return x + inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.outpud_dim,
            }
        )
        return config


class VectorQuantizationLayer(layers.Layer):
    def __init__(self, num_vectors, vector_dimension, beta=None, **kwargs) -> None:
        """Implementation of the Vector Quantization Layer from the original
        paper https://arxiv.org/abs/1711.00937


        Args:
            num_vectors (int): The number of encoding venctors for the codebook.
            vector_dimension (int): The dimension of the vectors.
            beta (_type_, float): The multiplicative parameter for the commitment loss term.
                Defaults to None.
        """
        super().__init__(**kwargs)

        self.num_vectors = num_vectors
        self.vector_dimension = vector_dimension

        if beta is not None:
            self.beta = beta
        else:
            self.beta = 0.25

        e_init = tf.random_uniform_initializer()
        self.embedding = tf.Variable(
            initial_value=e_init(
                shape=(self.vector_dimension, self.num_vectors), dtype="float32"
            ),
            trainable=True,
            name="embeddings",
        )  # C x K

    def get_indices(self, flatten_inputs):

        # flatten inputs is NxC dimensional, embedding is CxK dimensional
        # similarity will be an NxK dimensional matrix
        similarity = tf.matmul(flatten_inputs, self.embedding)

        # here we calculate the L2 distance
        distances = (
            tf.reduce_sum(flatten_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embedding**2, axis=0)
            - 2 * similarity
        )

        encoding_indices = tf.argmin(distances, axis=1)

        return encoding_indices

    def call(self, x):

        input_shape = tf.shape(x)
        flatten = tf.reshape(x, [-1, self.vector_dimension])

        encoding_indices = self.get_indices(flatten)  # N,
        encodings = tf.one_hot(encoding_indices, self.num_vectors)  # NxK
        quantized = tf.matmul(encodings, self.embedding, transpose_b=True)  # NxC

        quantized = tf.reshape(quantized, input_shape)  # B, H, W, C

        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # during the forward pass the two input terms cancel out (x and - x)
        # at backprop time, the stop gradient will exclude (quantized - x)
        # from the graph, so the gradient of quantized is actually copied to inputs
        # this is the implementation of the straight-through pass of VQ-VAE paper
        quantized = x + tf.stop_gradient(quantized - x)

        return quantized


class SinusoidalEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        embedding_max_frequency=1000.0,
        embedding_min_frequency=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.embedding_max_frequency = embedding_max_frequency
        self.embedding_min_frequency = embedding_min_frequency

        self.frequencies = tf.exp(
            tf.linspace(
                tf.math.log(embedding_min_frequency),
                tf.math.log(embedding_max_frequency),
                self.embed_dim // 2,
            )
        )
        self.angular_speeds = 2.0 * np.pi * self.frequencies

    def call(self, x):
        x = tf.concat(
            [tf.sin(self.angular_speeds * x), tf.cos(self.angular_speeds * x)], axis=3
        )

        return x
