import tensorflow as tf
import math


def wing_loss(x, w=10.0, epsilon=2.0):
    """
    Arguments:
        x: a float tensor with shape [batch_size].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [batch_size].
    """
    with tf.name_scope('wing_loss'):
        c = w * (1.0 - math.log(1.0 + w/epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.log(1.0 + absolute_x/epsilon),
            absolute_x - c
        )
        return losses
