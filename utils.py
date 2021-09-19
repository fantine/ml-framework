import tensorflow as tf


def random_crop(inputs, crop_shape):
  inputs = tf.image.random_crop(inputs, crop_shape)
  return inputs


def central_crop(inputs, crop_shape):
  if len(inputs.shape) != len(crop_shape):
    raise ValueError('Input shape and crop shape must be of same length.')
  offsets = [
      (inputs.shape[i] - crop_shape[i]) // 2 for i in range(len(crop_shape))]
  indices = tuple(
      slice(offsets[i], offsets[i] + crop_shape[i])
      for i in range(len(crop_shape))
  )
  return inputs[indices]
