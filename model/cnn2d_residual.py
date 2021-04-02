from tensorflow import keras

from ml_framework.model import base
from ml_framework.model import utils


class CNN2DResidual(base.ClassificationModel):

  def get_input_shape(self):
    h = self.hparams
    return (h.height, h.width, h.channels)

  @staticmethod
  def create_model(input_shape, hparams):
    layer_filters = utils.get_model_layers(
        hparams.network_depth,
        hparams.num_filters,
        hparams.filter_increase_mode,
    )

    regularizer = utils.get_regularizer(
        hparams.regularizer, hparams.regularizer_weight)

    inputs = keras.Input(shape=input_shape)
    x = residual_encoder(inputs, layer_filters, regularizer, hparams.batchnorm)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(hparams.filter_multiplier * layer_filters[-1])(x)
    if hparams.dense_dropout > 0:
      x = keras.layers.Dropout(hparams.dense_dropout)(x)
    if hparams.activation == 0:
      x = keras.layers.ReLU()(x)
    elif hparams.activation == 1:
      x = keras.layers.LeakyReLU()(x)
    else:
      raise NotImplementedError('Unsupported activation layer type.')
    x = keras.layers.Dense(hparams.out_channels)(x)
    return keras.Model(inputs=inputs, outputs=x)


def residual_encoder(inputs, layer_filters, regularizer, batchnorm):
  """Performs a series of downsamples to the input.

  Args:
    inputs: Input to the encoder.
    layer_filters: List of filters for successive layers.
    regularizer: Regularizer.
    batchnorm: Whether or not to use batch normalization.

  Returns:
    Output of the residual encoder.
  """
  x = keras.layers.Conv2D(
      filters=layer_filters[0], kernel_size=3, padding='same', use_bias=False,
      kernel_regularizer=regularizer)(inputs)
  if batchnorm == 1:
    x = keras.layers.BatchNormalization()(x)
  x = keras.layers.LeakyReLU()(x)
  x = keras.layers.Conv2D(
      filters=layer_filters[0], kernel_size=3, padding='same', use_bias=False,
      kernel_regularizer=regularizer)(x)

  shortcut = keras.layers.Conv2D(
      filters=layer_filters[0], kernel_size=1, padding='same', use_bias=False,
      kernel_regularizer=regularizer)(inputs)
  if batchnorm == 1:
    shortcut = keras.layers.BatchNormalization()(shortcut)
  x = shortcut + x

  for filters in layer_filters[1:]:
    x = utils.residual_block_2d(x, filters=[filters, filters], strides=[2, 1],
                                regularizer=regularizer, batchnorm=batchnorm)
  return x
