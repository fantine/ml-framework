from tensorflow import keras

from ml_framework.model import base
from ml_framework.model import utils


class CNN1DModular(base.ClassificationModel):

  def get_input_shape(self):
    h = self.hparams
    return (h.height, h.channels)

  @staticmethod
  def create_model(input_shape, hparams):
    layer_filters = utils.get_model_layers(
        hparams.network_depth,
        hparams.num_filters,
        hparams.filter_increase_mode,
    )

    regularizer = utils.get_regularizer(
        hparams.regularizer, hparams.regularizer_weight)

    model = keras.Sequential([keras.layers.Input(input_shape)])
    for filters in layer_filters:
      model.add(keras.layers.Conv1D(
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
        model.add(keras.layers.MaxPooling1D(2, padding='same'))
      elif hparams.downsampling == 1:  # Convolution downsample
        model.add(keras.layers.Conv1D(
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
