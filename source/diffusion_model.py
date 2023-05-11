import tensorflow as tf
from .denoiser import get_unet, get_swin_unet
from .losses import l1_loss
import matplotlib.pyplot as plt


class LatentDiffusionModel(tf.keras.models.Model):
    def __init__(
        self,
        input_shape,
        min_signal_rate=0.02,
        max_signal_rate=0.98,
        batch_size=8,
        **kwargs
    ):
        super().__init__()

        self._input_shape = input_shape
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        self.depths = kwargs.get("depths", [2, 8, 2])
        self.initial_dim = kwargs.get("initial_dim", 96)
        self.output_classes = kwargs.get("output_classes", 3)
        self.latent_image_shape = kwargs.get("latent_image_shape", (32, 32, 3))
        self.batch_size = batch_size

        self.vqgan = tf.keras.models.load_model("saved_model/vq_gan_256_model")

        self.encoder = self.vqgan.vq_vae.encoder
        self.decoder = self.vqgan.vq_vae.decoder
        self.quantization = self.vqgan.vq_vae.vq_layer

        mean = kwargs.get("mean", 0.0)
        variance = kwargs.get("variance", 1.0)

        self.normalization = tf.keras.layers.Normalization(mean=mean, variance=variance)

        self.denoiser = get_swin_unet(
            input_shape=self.latent_image_shape,
            depths=self.depths,
            initial_dim=self.initial_dim,
            output_classes=self.output_classes,
        )

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = tf.keras.metrics.Mean(name="noise_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.reconstruction_loss_tracker]

    def denormalize(self, inputs):

        inputs = self.normalization.mean + inputs * self.normalization.variance**0.5
        return tf.clip_by_value(inputs, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):

        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates):

        pred_noises = self.denoiser([noisy_images, noise_rates**2], training=True)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):

        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise

        for step in range(diffusion_steps):

            noisy_images = next_noisy_images

            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates
            )

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )

            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images

    def generate(self, num_images, diffusion_steps):

        initial_noise = tf.random.normal(
            shape=(
                num_images,
                *self.latent_image_shape,
            )
        )

        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)

        quantized = self.quantization(generated_images)
        reconstructed = self.decoder(quantized, training=False)

        return reconstructed

    def train_step(self, images):

        images = self.encoder(images, training=False)
        images = self.normalization(images)
        noises = tf.random.normal(shape=(self.batch_size, *self.latent_image_shape))

        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1),
            minval=0.0,
            maxval=1.0,
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = images * signal_rates + noises * noise_rates

        with tf.GradientTape() as tape:

            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates
            )

            noise_loss = l1_loss(noises, pred_noises)
            image_loss = l1_loss(images, pred_images)

        gradients = tape.gradient(noise_loss, self.denoiser.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.denoiser.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.reconstruction_loss_tracker.update_state(image_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):

        images = self.encoder(images, training=False)
        images = self.normalization(images)
        noises = tf.random.normal(shape=(self.batch_size, *self.latent_image_shape))

        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1),
            minval=0.0,
            maxval=1.0,
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = images * signal_rates + noises * noise_rates

        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates)

        noise_loss = l1_loss(noises, pred_noises)
        image_loss = l1_loss(images, pred_images)

        self.noise_loss_tracker.update_state(noise_loss)
        self.reconstruction_loss_tracker.update_state(image_loss)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(
        self, epoch=None, logs=None, num_rows=3, num_cols=6, diffusion_steps=20
    ):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
