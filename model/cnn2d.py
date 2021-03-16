import tensorflow as tf
from tensorflow import keras

from ml_framework.model import base


class CNN2DExample(base.ClassificationModel):

  def get_input_shape(self):
    h = self.hparams
    return (h.height, h.width, h.channels)

  @staticmethod
  def create_model(input_shape, hparams):
    return keras.Sequential([
        keras.layers.Input(input_shape, name='inputs'),
        keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(1024),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1)
    ])


class CNN2DModular(base.ClassificationModel):

  def get_input_shape(self):
    h = self.hparams
    return (h.height, h.width, h.channels)

  @staticmethod
  def create_model(input_shape, hparams):
    layer_filters = get_model_layers(
        hparams.network_depth,
        hparams.num_filters,
        hparams.filter_increase_mode,
    )

    regularizer = get_regularizer(
        hparams.regularizer, hparams.regularizer_weight)

    model = keras.Sequential([keras.layers.Input(input_shape)])
    for filters in layer_filters:
      model.add(keras.layers.Conv2D(
          filters, 3, padding='same', kernel_regularizer=regularizer))
      if hparams.batchnorm == 1:
        model.add(keras.layers.BatchNormalization())
      if hparams.activation == 0:
        model.add(keras.layers.ReLU())
      elif hparams.activation == 1:
        model.add(keras.layers.LeakyReLU())
      else:
        raise NotImplementedError('Unsupported activation layer type.')
      if hparams.conv_dropout > 0:
        model.add(keras.layers.Dropout(hparams.conv_dropout))
      if hparams.downsampling == 0:  # MaxPool
        model.add(keras.layers.MaxPooling2D(2, padding='same'))
      elif hparams.downsampling == 1:  # Convolution downsample
        model.add(keras.layers.Conv2D(
            filters, 3, strides=2, padding='same',
            kernel_regularizer=regularizer))
      else:
        raise NotImplementedError('Unsupported downsampling layer type.')

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        hparams.filter_multiplier * layer_filters[-1]))
    if hparams.dense_dropout > 0:
      model.add(keras.layers.Dropout(hparams.dense_dropout))
    if hparams.activation == 0:
      model.add(keras.layers.ReLU())
    elif hparams.activation == 1:
      model.add(keras.layers.LeakyReLU())
    else:
      raise NotImplementedError('Unsupported activation layer type.')

    model.add(keras.layers.Dense(hparams.out_channels))
    return model


def get_model_layers(depth, num_filters=16, increase_mode=1):
  if increase_mode == 1:  # Linear increase
    layers = [int(num_filters * (1 + i)) for i in range(depth)]
  if increase_mode == 2:  # Doubling every other block
    layers = [int(num_filters * 2**(i // 2)) for i in range(depth)]
  if increase_mode == 3:  # Doubling every block
    layers = [int(num_filters * 2**(i)) for i in range(depth)]
    if depth > 1:
      layers[-1] //= 2
  return layers


def get_regularizer(regularizer=0, reg_weight=1e-4):
  if regularizer == 0:
    return None
  if regularizer == 1:
    return keras.regularizers.l1(reg_weight)
  if regularizer == 2:
    return keras.regularizers.l2(reg_weight)
