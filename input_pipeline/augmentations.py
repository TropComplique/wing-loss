import tensorflow as tf
import cv2
import math


"""
`image` is assumed to be a float tensor with shape [height, width, 3],
it is a RGB image with pixel values in range [0, 1].
`box` is a float tensor with shape [4].
`landmarks` is a float tensor with shape [num_landmarks, 2].
"""


def random_rotation(image, box, landmarks, max_angle=10):
    with tf.name_scope('random_rotation'):
        # get a random angle
        max_angle_radians = max_angle*(math.pi/180.0)
        theta = tf.random_uniform(
            [], minval=-max_angle_radians,
            maxval=max_angle_radians, dtype=tf.float32
        )
    
        image_h = tf.to_float(tf.shape(image)[0])
        image_w = tf.to_float(tf.shape(image)[1])
        center = tf.stack([0.5*image_h, 0.5*image_w], axis=0)
        center = tf.reshape(center, [1, 2])
        
        scaler = tf.stack([image_h, image_w], axis=0)
        #box = box * scaler
    
        rotation = tf.stack([
            tf.cos(theta), tf.sin(theta),
            -tf.sin(theta), tf.cos(theta)
        ], axis=0)
        rotation_matrix2 = tf.reshape(rotation, [2, 2])
        
        rotation = tf.stack([
            tf.cos(theta), -tf.sin(theta),
            tf.sin(theta), tf.cos(theta)
        ], axis=0)
        rotation_matrix = tf.reshape(rotation, [2, 2])

        # rotate box and landmarks
        #box = tf.Print(box, [box, tf.reshape(box, [2, 2])])
        box = tf.matmul(tf.reshape(box, [2, 2])*scaler - center, rotation_matrix) + center
        box = box/scaler
        box = tf.reshape(box, [4])
        #box = tf.clip_by_value(box, 0.0, 1.0)
        landmarks = tf.matmul(landmarks*scaler - center, rotation_matrix) + center
        # note: landmark and box coordinates are (y, x) not (x, y)
        landmarks = landmarks/scaler
        
        
        translate = center - tf.matmul(center, rotation_matrix2)
        translate_y, translate_x = tf.unstack(tf.squeeze(translate, axis=0), axis=0)
        # rotate image
        transform = tf.stack([tf.cos(theta), tf.sin(theta), translate_x, -tf.sin(theta), tf.cos(theta), translate_y, 0.0, 0.0], axis=0)
        image = tf.contrib.image.transform(image, transform, interpolation='BILINEAR')

        return image, box, landmarks


def random_gaussian_blur(image, probability=0.3, kernel_size=3):

    def blur(image):
        image = (image*255.0).astype('uint8')
        image = cv2.blur(image, (kernel_size, kernel_size))
        return (image/255.0).astype('float32')

    with tf.name_scope('random_gaussian_blur'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(
            do_it,
            lambda: tf.py_func(blur, [image], tf.float32, stateful=False),
            lambda: image
        )
        return image


def random_color_manipulations(image, probability=0.5, grayscale_probability=0.1):

    def manipulate(image):
        # intensity and order of this operations are kinda random,
        # so you will need to tune this for you problem
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def to_grayscale(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image

    with tf.name_scope('random_color_manipulations'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: manipulate(image), lambda: image)

    with tf.name_scope('to_grayscale'):
        make_gray = tf.less(tf.random_uniform([]), grayscale_probability)
        image = tf.cond(make_gray, lambda: to_grayscale(image), lambda: image)

    return image


def random_flip_left_right(image, landmarks):

    def flip(image, landmarks):
        flipped_image = tf.image.flip_left_right(image)
        y, x = tf.unstack(landmarks, axis=1)
        flipped_x = tf.subtract(1.0, x)
        flipped_landmarks = tf.stack([y, flipped_x], axis=1)
        
        # landmarks:
        # left_eye right_eye
        # nose
        # left_mouth right_mouth
        correct_order = tf.constant([1, 0, 2, 4, 3], dtype=tf.int32)
        flipped_landmarks = tf.gather(flipped_landmarks, correct_order)

        return flipped_image, flipped_landmarks

    with tf.name_scope('random_flip_left_right'):
        do_it = tf.less(tf.random_uniform([]), 0.5)
        image, landmarks = tf.cond(do_it, lambda: flip(image, landmarks), lambda: (image, landmarks))
        return image, landmarks


def random_pixel_value_scale(image, minval=0.9, maxval=1.1, probability=0.5):
    """This function scales each pixel independently of the other ones.

    Arguments:
        image: a float tensor with shape [height, width, 3],
            an image with pixel values varying between [0, 1].
        minval: a float number, lower ratio of scaling pixel values.
        maxval: a float number, upper ratio of scaling pixel values.
        probability: a float number.
    Returns:
        a float tensor with shape [height, width, 3].
    """
    def random_value_scale(image):
        color_coefficient = tf.random_uniform(
            tf.shape(image), minval=minval,
            maxval=maxval, dtype=tf.float32
        )
        image = tf.multiply(image, color_coefficient)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    with tf.name_scope('random_pixel_value_scale'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: random_value_scale(image), lambda: image)
        return image


def random_box_jitter(box, landmarks, ratio=0.05):
    """Randomly jitter bounding boxes.

    Arguments:
        box: a float tensor with shape [4].
        landmarks: a float tensor with shape [num_landmarks, 2].
        ratio: a float number.
            The ratio of the box width and height that the corners can jitter.
            For example if the width is 100 pixels and ratio is 0.05,
            the corners can jitter up to 5 pixels in the x direction.
    Returns:
        a float tensor with shape [4].
    """
    with tf.name_scope('random_box_jitter'):

        y, x = tf.unstack(landmarks, axis=1)
        ymin, ymax = tf.reduce_min(y), tf.reduce_max(y)
        xmin, xmax = tf.reduce_min(x), tf.reduce_max(x)
        # we want to keep landmarks inside the new distorted box

        ymin2, xmin2, ymax2, xmax2 = tf.unstack(box, axis=0)
        box_height, box_width = ymax2 - ymin2, xmax2 - xmin2

        # it is assumed that initially
        # all landmarks were inside the box
        ymin3 = tf.random_uniform(
            [], minval=ymin2 - box_height * ratio,
            maxval=ymin, dtype=tf.float32
        )
        xmin3 = tf.random_uniform(
            [], minval=xmin2 - box_width * ratio,
            maxval=xmin, dtype=tf.float32
        )
        ymax3 = tf.random_uniform(
            [], minval=ymax,
            maxval=ymax2 + box_height * ratio,
            dtype=tf.float32
        )
        xmax3 = tf.random_uniform(
            [], minval=xmax,
            maxval=xmax2 + box_width * ratio,
            dtype=tf.float32
        )
        distorted_box = tf.stack([ymin3, xmin3, ymax3, xmax3], axis=0)
        distorted_box = tf.clip_by_value(distorted_box, 0.0, 1.0)
        return distorted_box
