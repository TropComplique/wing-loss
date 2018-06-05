import tensorflow as tf
import json
from model import model_fn


"""The purpose of this script is to export a savedmodel."""


CONFIG = 'config.json'
OUTPUT_FOLDER = 'export/run00'
GPU_TO_USE = '0'

tf.logging.set_verbosity('INFO')
params = json.load(open(CONFIG))
WIDTH, HEIGHT = params['image_size']

config = tf.ConfigProto()
config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'],
    session_config=config
)
estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)


def serving_input_receiver_fn():
    images = tf.placeholder(dtype=tf.uint8, shape=[None, HEIGHT, WIDTH, 3], name='image')
    features = tf.transpose(tf.to_float(images)*(1.0/255.0), perm=[0, 3, 1, 2])
    return tf.estimator.export.TensorServingInputReceiver(features=features, receiver_tensors={'images': images})


estimator.export_savedmodel(
    OUTPUT_FOLDER, serving_input_receiver_fn
)
