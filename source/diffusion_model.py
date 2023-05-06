import tensorflow as tf
from denoiser import get_unet


class LatentDiffusionModel(tf.keras.models.Model):
    def __init__(self, image_size, **kwargs):
        super().__init__()

        self.image_size = image_size
        self.depths = kwargs.get("depths", [4, 4, 8])
