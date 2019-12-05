# Authors: Thalles, Felipe, Illiana
# Define data pre-processing techniques used during training
import tensorflow as tf


def tf_record_parser(record):
    keys_to_features = {
        "image_raw": tf.io.FixedLenFeature((), tf.string, default_value=""),
        "height": tf.io.FixedLenFeature((), tf.int64),
        "width": tf.io.FixedLenFeature((), tf.int64),
        "label": tf.io.FixedLenFeature((), tf.int64),
        "depth": tf.io.FixedLenFeature((), tf.int64)
    }

    features = tf.io.parse_single_example(record, keys_to_features)

    image = tf.io.decode_raw(features['image_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    # reshape input and annotation images
    image = tf.reshape(image, (height, width, depth), name="image_reshape")
    return image, label


def normalizer(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return image, label


def clip_image(image, label):
    return tf.clip_by_value(image, 0.0, 1.0), label


def random_flip_left_right(image, label):
    return tf.image.random_flip_left_right(image), label


def random_flip_up_down(image, label):
    return tf.image.random_flip_up_down(image), label


def random_rotation(image, label):
    return tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)), label


def random_brightness(image, label):
    return tf.image.random_brightness(image, max_delta=32. / 255.), label


def random_contrast(image, label):
    return tf.image.random_contrast(image, lower=0.7, upper=1.3), label


def random_saturation(image, label):
    return tf.image.random_saturation(image, lower=0.7, upper=1.3), label


def random_hue(image, label):
    return tf.image.random_hue(image, max_delta=0.2), label
