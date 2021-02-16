import tensorflow as tf


def random_crop(inputs, crop_shape):
  inputs = tf.image.random_crop(inputs, crop_shape)
  return inputs
