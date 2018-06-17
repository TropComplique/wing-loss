import tensorflow as tf
import tensorflow.contrib.slim as slim


BATCH_NORM_MOMENTUM = 0.91


def network(images, is_training, num_landmarks):
    """
    Arguments:
        images: a float tensor with shape [batch_size, 3, height, width],
            a batch of RGB images with pixels values in the range [0, 1].
        is_training: a boolean.
        num_landmarks: an integer.
    Returns:
        a float tensor with shape [batch_size, 2 * num_landmarks].
    """

    def preprocess(images):
        """Transform images before feeding them to the network."""
        return (2.0*images) - 1.0

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=1, center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM, epsilon=1e-4,
            training=is_training, fused=True,
            name='batch_norm'
        )
        return x

    def prelu(x):
        with tf.variable_scope('prelu'):
            in_channels = x.shape.as_list()[1]
            shape = [1, in_channels, 1, 1] if len(x.shape) == 4 else [1, in_channels]
            alpha = tf.get_variable(
                'alpha', shape,
                initializer=tf.constant_initializer(0.1),
                dtype=tf.float32
            )
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    with tf.name_scope('standardize_input'):
        x = preprocess(images)

    with tf.variable_scope('network'):
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu,#prelu,
            'normalizer_fn': batch_norm,
            'data_format': 'NCHW'
        }
        with slim.arg_scope([slim.conv2d], **params):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME', data_format='NCHW'):

                num_filters = [32, 64, 128, 256, 512]
                for i, f in enumerate(num_filters, 1):
                    x = slim.conv2d(x, f, (3, 3), stride=1, scope='conv%d' % i)
                    #x = slim.conv2d(x, f, (3, 3), stride=2, scope='conv%d_downsample' % i)
                    x = slim.max_pool2d(x, (2, 2), scope='pool%d' % i)

        x = flatten(x)
        x = slim.fully_connected(
            x, 1024, activation_fn=tf.nn.relu,
            normalizer_fn=batch_norm, scope='fc1'
        )
        x = slim.fully_connected(
            x, 2 * num_landmarks, activation_fn=tf.sigmoid,
            normalizer_fn=None, scope='fc2'
        )
        x = tf.reshape(x, [-1, num_landmarks, 2])
        return x


def flatten(x):
    with tf.name_scope('flatten'):
        batch_size = tf.shape(x)[0]
        channels, height, width = x.shape.as_list()[1:]
        x = tf.reshape(x, [batch_size, channels * height * width])
        return x
