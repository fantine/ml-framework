from tensorflow import keras

from ml_framework.model import base


class CNN1DExample(base.ClassificationModel):

  def get_input_shape(self):
    h = self.hparams
    return (h.height, h.channels)

  @staticmethod
  def create_model(input_shape, hparams):
    return keras.Sequential([
        keras.layers.Input(input_shape),
        keras.layers.Conv1D(16, 3, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv1D(16, 3, strides=2, padding='same'),
        keras.layers.Conv1D(32, 3, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv1D(32, 3, strides=2, padding='same'),
        keras.layers.Conv1D(64, 3, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv1D(64, 3, strides=2, padding='same'),
        keras.layers.Conv1D(128, 3, padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv1D(128, 3, strides=2, padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(1024),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dense(1),
    ])
