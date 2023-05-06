import tensorflow as tf


def discriminator_loss(real_output, generated_output):

    ones = tf.ones_like(real_output)
    zeros = tf.zeros_like(generated_output)

    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(ones, real_output)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        zeros, generated_output
    )

    return real_loss + generated_loss

def l1_loss(y_true, y_pred):

    loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    return loss

def generator_loss(y_true, y_pred, disc_output, beta=100):

    ones = tf.ones_like(disc_output)

    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(ones, disc_output)

    mae = l1_loss(y_true, y_pred)

    total_loss = gan_loss + beta * mae

    return total_loss

