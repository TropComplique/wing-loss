import tensorflow as tf
from .augmentations import random_color_manipulations,\
    random_flip_left_right, random_pixel_value_scale, \
    random_gaussian_blur, random_rotation, random_box_jitter


SHUFFLE_BUFFER_SIZE = 20000
NUM_THREADS = 8
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR


class Pipeline:
    def __init__(self, filenames, batch_size, image_size, num_landmarks,
                 repeat=False, shuffle=False, augmentation=False):
        """
        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            batch_size: an integer.
            image_size: a list with two integers [width, height],
                images of this size will be in a batch
            num_landmarks: an integer.
            repeat: a boolean, whether repeat indefinitely.
            shuffle: whether to shuffle the dataset.
            augmentation: whether to do data augmentation.
        """
        self.image_width, self.image_height = image_size
        self.augmentation = augmentation
        self.batch_size = batch_size

        assert num_landmarks == 5
        self.num_landmarks = num_landmarks

        def get_num_samples(filename):
            return sum(1 for _ in tf.python_io.tf_record_iterator(filename))

        num_examples = 0
        for filename in filenames:
            num_examples_in_file = get_num_samples(filename)
            assert num_examples_in_file > 0
            num_examples += num_examples_in_file
        self.num_examples = num_examples
        assert self.num_examples > 0

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=num_shards)
        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=batch_size)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(None if repeat else 1)
        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_THREADS)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def _parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. (optionally) Augments it.

        Returns:
            image: a float tensor with shape [image_height, image_width, 3],
                an RGB image with pixel values in the range [0, 1].
            landmarks: a float tensor with shape [num_landmarks, 2].
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'ymin': tf.FixedLenFeature([], tf.float32),
            'xmin': tf.FixedLenFeature([], tf.float32),
            'ymax': tf.FixedLenFeature([], tf.float32),
            'xmax': tf.FixedLenFeature([], tf.float32),
            'landmarks': tf.FixedLenFeature([2 * self.num_landmarks], tf.float32)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get image
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # now pixel values are scaled to [0, 1] range

        # get face box, it must be in from-zero-to-one format
        box = tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=0)
        box = tf.clip_by_value(tf.to_float(box), clip_value_min=0.0, clip_value_max=1.0)

        # get facial landmarks, they must be in from-zero-to-one format
        landmarks = tf.to_float(parsed_features['landmarks'])
        landmarks = tf.reshape(landmarks, [self.num_landmarks, 2])
        landmarks = tf.clip_by_value(landmarks, clip_value_min=0.0, clip_value_max=1.0)
        # it assumed that landmarks are inside the bounding box (or on the edges)

        if self.augmentation:
            image, landmarks = self._augmentation_fn(image, box, landmarks)
        else:
            image, landmarks = crop(image, landmarks, box)
            image = tf.image.resize_images(
                image, [self.image_height, self.image_width],
                method=RESIZE_METHOD
            )

        return image, landmarks

    def _augmentation_fn(self, image, box, landmarks):
        # there are a lot of hyperparameters here,
        # you will need to tune them all, haha
        image, box, landmarks = random_rotation(image, box, landmarks, max_angle=5)
        box = random_box_jitter(box, landmarks, ratio=0.025)
        image, landmarks = crop(image, landmarks, box)
        image = tf.image.resize_images(
            image, [self.image_height, self.image_width],
            method=RESIZE_METHOD
        )
        image = random_color_manipulations(image, probability=0.1, grayscale_probability=0.01)
        image = random_pixel_value_scale(image, minval=0.85, maxval=1.15, probability=0.1)
        image = random_gaussian_blur(image, probability=0.1, kernel_size=4)
        image, landmarks = random_flip_left_right(image, landmarks)
        return image, landmarks


def crop(image, landmarks, box):
    """
    Crops the image to the box.
    It also adds some margin.
    Finally, it transforms coordinates of the landmarks.
    """
    image_h = tf.to_float(tf.shape(image)[0])
    image_w = tf.to_float(tf.shape(image)[1])
    scaler = tf.stack([image_h, image_w, image_h, image_w], axis=0)
    box = box * scaler
    ymin, xmin, ymax, xmax = tf.unstack(box, axis=0)

    h, w = ymax - ymin, xmax - xmin
    margin_y, margin_x = h / 6.0, w / 6.0  # 6.0 here is a hyperparameter

    ymin, xmin = ymin - 0.5 * margin_y, xmin - 0.5 * margin_x
    ymax, xmax = ymax + 0.5 * margin_y, xmax + 0.5 * margin_x
    ymin = tf.clip_by_value(ymin, 0.0, image_h)
    xmin = tf.clip_by_value(xmin, 0.0, image_w)
    ymax = tf.clip_by_value(ymax, 0.0, image_h)
    xmax = tf.clip_by_value(xmax, 0.0, image_w)

    # for some reason box width or height sometimes becomes zero,
    # but it happens very very rarely
    h, w = tf.to_int32(ymax - ymin), tf.to_int32(xmax - xmin)
    box_is_okay = tf.greater(h * w, 0)

    def do_it(image, landmarks):
        image = tf.image.crop_to_bounding_box(
            image, tf.to_int32(ymin), tf.to_int32(xmin), h, w
        )
        # translate coordinates of the landmarks
        shift = tf.stack([ymin/(ymax - ymin), xmin/(xmax - xmin)], axis=0)
        scaler = tf.stack([image_h/(ymax - ymin), image_w/(xmax - xmin)], axis=0)
        landmarks = (landmarks * scaler) - shift
        return image, landmarks

    image, landmarks = tf.cond(
        box_is_okay,
        lambda: do_it(image, landmarks),
        lambda: (image, landmarks)
    )

    landmarks = tf.clip_by_value(landmarks, 0.0, 1.0)
    return image, landmarks
