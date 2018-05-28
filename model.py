import tensorflow as tf
import re

from network import network
from loss import wing_loss


MOMENTUM = 0.9


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

    with tf.name_scope('evaluation_ops'):
        probabilities = tf.nn.softmax(logits, axis=1)
        predictions = tf.argmax(probabilities, axis=1)
        accuracy = tf.metrics.accuracy(labels, predictions)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['lr_boundaries'], params['lr_values'])
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM, use_nesterov=True)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    tf.summary.scalar('train_accuracy', accuracy[1])
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
