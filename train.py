import tensorflow as tf
import json
import os

from model import model_fn
from input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


params = json.load(open('config.json'))


def get_input_fn(is_training=True):

    image_size = params['image_size']
    data_dir = params['train_dataset'] if is_training else params['val_dataset']
    batch_size = params['batch_size']
    num_landmarks = params['num_landmarks']

    filenames = os.listdir(data_dir)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = sorted(filenames)
    filenames = [os.path.join(data_dir, n) for n in filenames]

    def input_fn():
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filenames, batch_size=batch_size, image_size=image_size, num_landmarks=num_landmarks,
                repeat=is_training, shuffle=is_training, augmentation=is_training,
            )
            features, labels = pipeline.get_batch()
        return features, labels

    return input_fn


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.visible_device_list = '1'

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'],
    session_config=config,
    save_summary_steps=100,
    save_checkpoints_secs=600,
    log_step_count_steps=100
)

train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)

estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)
train_spec = tf.estimator.TrainSpec(
    train_input_fn, max_steps=params['num_steps']
)
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None,
    start_delay_secs=1200, throttle_secs=1200
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
