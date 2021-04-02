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
