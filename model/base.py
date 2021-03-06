import abc
import logging
import os

import numpy as np
import tensorflow as tf

from ml_framework import metrics


class BaseModel():

  __metaclass__ = abc.ABCMeta

  def __init__(self, hparams, run_config):
    self._hparams = hparams
    self._run_config = run_config

  @property
  def hparams(self):
    return self._hparams

  @property
  def run_config(self):
    return self._run_config

  @abc.abstractmethod
  def get_model(self):
    return

  @abc.abstractmethod
  def get_optimizer(self):
    return

  @staticmethod
  @abc.abstractmethod
  def get_loss():
    return

  @staticmethod
  @abc.abstractmethod
  def get_metrics():
    return

  def train_and_evaluate(self, train_data, eval_data):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      model = self.get_model()
      model.compile(optimizer=self.get_optimizer(),
                    loss=self.get_loss(), metrics=self.get_metrics())
    model.summary()
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.hparams.job_dir, 'logs')),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.hparams.job_dir, 'ckpt'),
            save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=self.hparams.patience),
    ]
    model.fit(train_data, epochs=self.hparams.num_epochs,
              validation_data=eval_data, verbose=2, callbacks=callbacks,
              class_weight={0: 1., 1:self.hparams.pos_weight})

  def evaluate(self, eval_data):
    logging.info('Loading model from %s', self.hparams.job_dir)
    model = tf.keras.models.load_model(self.hparams.job_dir)
    model.compile(optimizer=self.get_optimizer(),
                  loss=self.get_loss(), metrics=self.get_metrics())
    model.evaluate(eval_data, verbose=2)
    logits = model.predict(eval_data)
    logits_file = os.path.join(self.hparams.job_dir, 'eval_logits.npy')
    logging.info('Writing predictions to %s', logits_file)
    with tf.io.gfile.GFile(logits_file, 'wb') as f:
      np.save(f, logits)

  def predict(self, test_datasets):
    logging.info('Loading model from %s', self.hparams.job_dir)
    model = tf.keras.models.load_model(self.hparams.job_dir)
    model.compile(optimizer=self.get_optimizer(),
                  loss=self.get_loss(), metrics=self.get_metrics())
    for key, test_dataset in test_datasets:
      logits = model.predict(test_dataset)
      logits_file = '{}_logits.npy'.format(key)
      logging.info('Writing logits to %s', logits_file)
      with tf.io.gfile.GFile(logits_file, 'wb') as f:
        np.save(f, logits)


class ClassificationModel(BaseModel):

  @abc.abstractmethod
  def get_input_shape(self):
    raise NotImplementedError

  @staticmethod
  @abc.abstractmethod
  def create_model(input_shape, hparams):
    raise NotImplementedError

  def get_optimizer(self):
    return tf.keras.optimizers.Adam(self.hparams.learning_rate)

  @staticmethod
  def get_loss():
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)

  @staticmethod
  def get_metrics():
    return [tf.keras.metrics.BinaryAccuracy(),
            metrics.AUC(from_logits=True),
            metrics.Precision(from_logits=True),
            metrics.Recall(from_logits=True),
            ]

  def get_model(self):
    return self.create_model(self.get_input_shape(), self.hparams)
