import tensorflow as tf
from typing import Callable
import numpy as np


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        decay_schedule_fn: Callable,
        start_lr: float,
        target_lr: float,
        warmup_steps: int,
        name: str = None,
    ):
        """Wrapper function to add a warmup start to a tensorflow decay schedule
        Args:
            decay_schedule_fn (Callable): Tensorflow LearningRateSchedule function or
                a custom Schedule with a __call__ method that accepts a step
            start_lr (float): Initial learning rate
            target_lr (float): Learning rate to reach at the end of the warmup
            warmup_steps (int): Number of warmup steps
            name (str, optional): Name of the object. Defaults to None.
        """
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_lr = (self.target_lr - self.start_lr) * (
                global_step_float / warmup_steps_float
            ) + self.start_lr

            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_lr,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "start_lr": self.start_lr,
            "decay_schedule_fn": self.decay_schedule_fn,
            "target_lr": self.target_lr,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }

