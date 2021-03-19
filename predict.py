import argparse
import logging
import sys
from typing import Text

import tensorflow as tf

from ml_framework import input_utils
from ml_framework import model


def _get_model(model_name: Text):
  """Gets the machine learning model corresponding to `model_name`."""
  return getattr(model, model_name)


def _parse_known_arguments(argv):
  """Parses command-line arguments."""

  parser = argparse.ArgumentParser()

  # I/O arguments
  parser.add_argument(
      '--overlap',
      help='Overlap.',
      default=0.5)
  parser.add_argument(
      '--job_dir',
      help='Directory to which to write model checkpoints and exports.',
      required=True)
  parser.add_argument(
      '--test_file',
      help='Test data file pattern.',
      required=True)
  parser.add_argument(
      '--log_level',
      help='Log level.',
      default='INFO')

  # Model definition arguments
  parser.add_argument(
      '--model',
      help='Name of the ML model to run.',
      required=True)
  parser.add_argument(
      '--height',
      help='Input height.',
      type=int,
      required=True)
  parser.add_argument(
      '--width',
      help='Input width.',
      type=int,
      default=1)
  parser.add_argument(
      '--depth',
      help='Input depth.',
      type=int,
      default=1)
  parser.add_argument(
      '--channels',
      help='Number of input channels.',
      type=int,
      default=1)

  # Training and evaluation arguments
  parser.add_argument(
      '--learning_rate',
      help='Learning rate.',
      type=float,
      default=0.0001)
  parser.add_argument(
      '--batch_size',
      help='Batch size.',
      type=int,
      default=32)
  return parser.parse_known_args(argv)


def _get_run_config(hparams) -> tf.estimator.RunConfig:
  """Gets the TensorFlow run config as defined by `hparams`."""
  return tf.estimator.RunConfig(model_dir=hparams.job_dir)


def run_experiment(hparams):
  """Runs predictions on streaming data.

  Args:
      hparams: Machine learning model hyperparameters.
  """
  test_data = input_utils.get_streaming_data(hparams)
  run_config = _get_run_config(hparams)
  model_ = _get_model(hparams.model)(hparams, run_config)
  logging.info('Running model: %s', hparams.model)
  model_.predict(test_data)


def _set_logging(log_level: Text):
  """Sets the logging level to `log_level`."""
  logger = tf.get_logger()
  logger.setLevel(log_level)
  return


def main():
  """Runs the experiment."""
  hparams, _ = _parse_known_arguments(sys.argv[1:])
  _set_logging(hparams.log_level.upper())
  run_experiment(hparams)


if __name__ == '__main__':
  main()
