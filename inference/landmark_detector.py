import tensorflow as tf
import numpy as np


class KeypointDetector:
    def __init__(self, model_path, gpu_memory_fraction=0.25, visible_device_list='0'):
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='import')

        self.input_image = graph.get_tensor_by_name('import/images:0')
        self.output = graph.get_tensor_by_name('import/landmarks:0')

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)

    def __call__(self, images):
        """
        Arguments:
            images: a numpy uint8 array with shape [b, 64, 64, 3],
                that represents a batch of RGB images.
        Returns:
            a float numpy array of shape [b, 5, 2].

        Note that points coordinates are in the order: (y, x).
        Also coordinates are relative to the image (in the [0, 1] range).
        """
        landmarks = self.sess.run(self.output, feed_dict={self.input_image: images})
        return landmarks
