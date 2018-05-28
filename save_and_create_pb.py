import tensorflow as tf
import json
import argparse
from model import model_fn


"""The purpose of this script is to export a savedmodel."""


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output', type=str, default='export/run03'
    )
    return parser.parse_args()


tf.logging.set_verbosity('INFO')
ARGS = make_args()
params = json.load(open('config_reproduction.json'))
width, height = params['image_size']

config = tf.ConfigProto()
config.gpu_options.visible_device_list = '1'
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'],
    session_config=config
)
estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)


def serving_input_receiver_fn():
    images = tf.placeholder(dtype=tf.uint8, shape=[None, height, width, 3], name='image')

    # convert to float, normalize to [0, 1] range, convert to NCHW format
    features = tf.transpose(tf.to_float(images)*(1.0/255.0), perm=[0, 3, 1, 2])

    return tf.estimator.export.TensorServingInputReceiver(features=features, receiver_tensors={'images': images})


estimator.export_savedmodel(
    ARGS.output, serving_input_receiver_fn
)

import tensorflow as tf
import argparse

"""Create a .pb frozen inference graph from a SavedModel."""


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--saved_model_folder', type=str
    )
    parser.add_argument(
        '-o', '--output_pb', type=str, default='model.pb'
    )
    return parser.parse_args()


def main():

    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = '1'
        with tf.Session(graph=graph, config=config) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], ARGS.saved_model_folder)

            # output ops
            keep_nodes = ['embedding']

            input_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(),
                output_node_names=keep_nodes
            )
            output_graph_def = tf.graph_util.remove_training_nodes(
                input_graph_def, protected_nodes=keep_nodes
            )

            with tf.gfile.GFile(ARGS.output_pb, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


ARGS = make_args()
tf.logging.set_verbosity('INFO')
main()
