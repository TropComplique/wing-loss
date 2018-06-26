import tensorflow as tf

"""
For evaluation during the training I use NME(normalized mean error).
You can find it here:
https://arxiv.org/abs/1506.03799 (Pose-Invariant 3D Face Alignment)
But my version is slightly different.

It is assumed that num_landmarks = 5
and that they are in the following order:
[[lefteye_y lefteye_x]
 [righteye_y righteye_x]
 [nose_y nose_x]
 [leftmouth_y leftmouth_x]
 [rightmouth_y rightmouth_x]]
"""


def nme_metric_ops(labels, landmarks):
    """
    Arguments:
        labels, landmarks: a float tensors with shape [batch_size, num_landmarks, 2].
    Returns:
        two ops like in tf.metrics API.
    """
    norms = tf.norm(labels - landmarks, axis=2)
    mean_norm = tf.reduce_mean(norms, axis=1)  # shape [batch_size]
    eye_distance = tf.norm(labels[:, 0, :] - labels[:, 1, :], axis=1)  # shape [batch_size]

    values = mean_norm/tf.maximum(eye_distance, 1.0)
    mean, update_op = tf.metrics.mean(values)
    return mean, update_op
