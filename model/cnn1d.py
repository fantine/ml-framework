import tensorflow as tf

from trainer.model import base


class ExampleCNN1D(base.ClassificationModel):

  def get_input_shape(self):
    h = self.hparams
    return (h.height, h.channels)

  @staticmethod
  def model(input_shape, hparams):
    return tf.keras.Sequential([
        tf.keras.layers.Input(input_shape, name='inputs'),
        tf.keras.layers.Conv1D(16, 3, padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1D(16, 3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(32, 3, padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1D(32, 3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(64, 3, padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1D(64, 3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(128, 3, padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv1D(128, 3, strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(1),
    ])
