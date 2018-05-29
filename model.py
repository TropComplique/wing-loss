import tensorflow as tf

from network import network
from loss import wing_loss
from evaluation import get_nme_metric_ops


def model_fn(features, labels, mode, params, config):
    """
    This is a function for creating a tensorflow computational graph.
    The function is in the format required by tf.estimator.
    """

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    landmarks = network(features, is_training, num_landmarks=params['num_landmarks'])
    # (features are just a tensor of RGB images)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'landmarks': landmarks}
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )

    losses = wing_loss(landmarks - labels, w=params['w'], epsilon=params['epsilon'])
    loss = tf.reduce_mean(losses, axis=0)
    tf.losses.add_loss(loss)
    tf.summary.scalar('wing_loss', loss)

    # add L2 regularization
    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar('regularization_loss', regularization_loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'mean_absolute_error': tf.metrics.mean_absolute_error(labels, landmarks),
            'normalized mean error': get_nme_metric_ops(labels, landmarks)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    with tf.name_scope('evaluation_ops'):
        mae = tf.reduce_mean(tf.reduce_sum(tf.abs(labels - landmarks), axis=[1, 2]), axis=0)

    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.polynomial_decay(
            params['initial_learning_rate'], global_step,
            decay_steps=params['num_steps'],
            end_learning_rate=params['end_learning_rate'],
            power=1.0  # linear decay
        )
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    tf.summary.scalar('train_mae', mae)
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    """Add L2 regularization to all (or some) trainable kernel weights."""
    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )
    trainable_vars = tf.trainable_variables()
    kernels = [v for v in trainable_vars if 'weights' in v.name]
    for K in kernels:
        x = tf.multiply(weight_decay, tf.nn.l2_loss(K))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)
