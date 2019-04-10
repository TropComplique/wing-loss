import tensorflow as tf
import json
import os
from model import model_fn, RestoreMovingAverageHook
from input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


CONFIG = 'config.json'
GPU_TO_USE = '0'
params = json.load(open(CONFIG))


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
        return pipeline.dataset
    return input_fn


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.visible_device_list = GPU_TO_USE

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'],
    session_config=config,
    save_summary_steps=2000,
    save_checkpoints_secs=1800,
    log_step_count_steps=1000
)

train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)

estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)
train_spec = tf.estimator.TrainSpec(
    train_input_fn, max_steps=params['num_steps']
)
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, hooks=[RestoreMovingAverageHook(params['model_dir'])],
    start_delay_secs=3600, throttle_secs=3600
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
