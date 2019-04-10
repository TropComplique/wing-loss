import tensorflow as tf
import tensorflow.contrib.slim as slim


BATCH_NORM_MOMENTUM = 0.91
BATCH_NORM_EPSILON = 1e-3


def network(images, is_training, num_landmarks):
    """
    Arguments:
        images: a float tensor with shape [batch_size, height, width, 3],
            a batch of RGB images with pixels values in the range [0, 1].
        is_training: a boolean.
        num_landmarks: an integer.
    Returns:
        a float tensor with shape [batch_size, num_landmarks, 2],
        it has values in the range [0, 1].
    """

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            training=is_training, fused=True,
            name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('network'):
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm,
            'data_format': 'NHWC'
        }
        with slim.arg_scope([slim.conv2d], **params):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME', data_format='NHWC'):

                num_filters = [32, 64, 128, 256, 512]
                for i, f in enumerate(num_filters, 1):
                    x = slim.conv2d(x, f, (3, 3), stride=1, scope='conv%d' % i)
                    x = slim.max_pool2d(x, (2, 2), scope='pool%d' % i)

        x = flatten(x)
        x = slim.fully_connected(
            x, 1024, activation_fn=tf.nn.relu,
            normalizer_fn=None, scope='fc1'
        )
        x = slim.fully_connected(
            x, 2 * num_landmarks, activation_fn=None,
            normalizer_fn=None, scope='fc2',
            biases_initializer=tf.constant_initializer(0.5),
        )
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, num_landmarks, 2])
        return x


def flatten(x):
    with tf.name_scope('flatten'):
        batch_size = tf.shape(x)[0]
        height, width, channels = x.shape.as_list()[1:]
        x = tf.reshape(x, [batch_size, channels * height * width])
        return x


def prelu(x):
    """It is not used here."""
    with tf.variable_scope('prelu'):
        in_channels = x.shape[3].value
        alpha = tf.get_variable(
            'alpha', [in_channels],
            initializer=tf.constant_initializer(0.1),
            dtype=tf.float32
        )
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
