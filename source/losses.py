import tensorflow as tf

def l1_loss(y_true, y_pred):

    loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    return loss


