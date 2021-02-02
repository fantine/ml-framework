import argparse
import logging
import sys

import tensorflow as tf

from trainer import input_utils
from trainer import model


def _get_model(model_name):
  return getattr(model, model_name)


def _parse_arguments(argv):
  """Parse command-line arguments."""

  parser = argparse.ArgumentParser()

  # I/O arguments
  parser.add_argument(
      '--job_dir',
      help='Directory to which to write model checkpoints and exports.',
      required=True)
  parser.add_argument(
      '--train_file',
      help='Training data file pattern.',
      required=True)
  parser.add_argument(
      '--eval_file',
      help='Evaluation data file pattern.',
      required=True)
  parser.add_argument(
      '--compression_type',
      help='Compression type.',
      default='GZIP')
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
  parser.add_argument(
      '--label_height',
      help='Label height.',
      type=int,
      default=1)
  parser.add_argument(
      '--label_width',
      help='Label width.',
      type=int,
      default=1)
  parser.add_argument(
      '--label_depth',
      help='Label depth.',
      type=int,
      default=1)
  parser.add_argument(
      '--channels_out',
      help='Number of output channels.',
      type=int,
      default=1)

  # Training and evaluation arguments
  parser.add_argument(
      '--train_steps',
      help='Number of training steps.',
      type=int,
      default=1000)
  parser.add_argument(
      '--eval_steps',
      help='Number of evaluation steps.',
      type=int,
      default=1000)
  parser.add_argument(
      '--batch_size',
      help='Batch size.',
      type=int,
      default=32)
  parser.add_argument(
      '--num_epochs',
      help='Number of training epochs.',
      type=int,
      default=10)
  parser.add_argument(
      '--learning_rate',
      help='Learning rate.',
      type=float,
      default=0.0001)
  parser.add_argument(
      '--shuffle_buffer_size',
      help='Input data shuffle buffer size.',
      type=int,
      default=1000)

  # Model architecture arguments
  parser.add_argument(
      '--network_depth',
      help='Parameter to control the number of layers in the network.',
      type=int,
      default=3)
  parser.add_argument(
      '--num_filters',
      help='Parameter to control the number of filters in the first layer.',
      type=int,
      default=16)
  parser.add_argument(
      '--filter_increase_mode',
      help=('Parameter to control the increase in number of filters in'
            ' successive layers.'),
      type=int,
      default=1)
  parser.add_argument(
      '--activation',
      help='Parameter to control the type of activation layers.',
      type=int,
      default=0)
  parser.add_argument(
      '--downsampling',
      help='Parameter to control the type of downsampling layers.',
      type=int,
      default=0)
  parser.add_argument(
      '--batchnorm',
      help='Parameter to control whether or not to use batch normalization.',
      type=int,
      default=0)
  parser.add_argument(
      '--conv_dropout',
      help='Parameter to control the dropout rate in convolutional layers.',
      type=float,
      default=0.0)
  parser.add_argument(
      '--dense_dropout',
      help='Parameter to control the dropout rate in dense layers.'
      type=float,
      default=0.0)
  parser.add_argument(
      '--regularizer',
      help='Parameter to control the type of regularization.',
      type=int,
      default=0)
  parser.add_argument(
      '--regularizer_weight',
      help='Parameter to control the regularization weight.',
      type=float,
      default=0.0)
  return parser.parse_args(argv)


def _get_run_config(hparams):
  return tf.estimator.RunConfig(model_dir=hparams.job_dir)


def run_experiment(hparams):
  train_data = input_utils.get_dataset(hparams, tf.estimator.ModeKeys.TRAIN)
  eval_data = input_utils.get_dataset(hparams, tf.estimator.ModeKeys.EVAL)
  run_config = _get_run_config(hparams)
  model_ = _get_model(hparams.model)(hparams, run_config)
  logging.info('Running model: %s', hparams.model)
  model_.train_and_evaluate(train_data, eval_data)


def _set_logging(log_level):
  logger = tf.get_logger()
  logger.setLevel(log_level)
  return


def main():
  """Runs the experiment."""
  hparams = _parse_arguments(sys.argv[1:])
  _set_logging(hparams.log_level.upper())
  run_experiment(hparams)


if __name__ == '__main__':
  main()
