import tensorflow as tf

from trainer import utils


def _parse_function(example_proto, mode, input_shape, label_shape,
                    tfrecord_shape):
  keys_to_features = {'inputs': tf.io.FixedLenFeature([], tf.string)}
  if mode != tf.estimator.ModeKeys.PREDICT:
    keys_to_features['labels'] = tf.io.FixedLenFeature([], tf.string)

  parsed_example = tf.io.parse_example(example_proto, keys_to_features)
  inputs = tf.io.decode_raw(parsed_example['inputs'], tf.float32)
  inputs = tf.reshape(inputs, tfrecord_shape)

  if tfrecord_shape != input_shape:
    inputs = utils.random_crop(inputs, input_shape)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return inputs

  labels = tf.io.decode_raw(parsed_example['labels'], tf.float32)
  labels = tf.reshape(labels, label_shape)
  return inputs, labels


def _get_dataset(
    file_pattern,
    input_shape,
    label_shape,
    tfrecord_shape,
    batch_size,
    mode=tf.estimator.ModeKeys.TRAIN,
    compression_type=None,
    shuffle=True,
    shuffle_buffer_size=10000
):
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
  dataset = dataset.interleave(
      lambda x: tf.data.TFRecordDataset(
          x, compression_type=compression_type),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(
      lambda x: _parse_function(
          x, mode, input_shape, label_shape, tfrecord_shape),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


def get_input_shape(hparams):
  if hparams.depth == 1:
    if hparams.width == 1:
      return (hparams.height, hparams.channels)
    return (hparams.height, hparams.width, hparams.channels)
  return (hparams.height, hparams.width, hparams.depth, hparams.channels)


def get_label_shape(hparams):
  if hparams.label_depth == 1:
    if hparams.label_width == 1:
      return (hparams.label_height,)
    return (hparams.label_height, hparams.label_width)
  return (hparams.label_height, hparams.label_width, hparams.label_depth)


def get_tfrecord_shape(hparams):
  if hparams.tfrecord_height == -1:
    return get_input_shape(hparams)
  if hparams.tfrecord_depth == 1:
    if hparams.tfrecord_width == 1:
      return (hparams.tfrecord_height, hparams.channels)
    return (hparams.tfrecord_height, hparams.tfrecord_width, hparams.channels)
  return (hparams.tfrecord_height, hparams.tfrecord_width,
          hparams.tfrecord_depth, hparams.channels)


def get_dataset(hparams, mode):
  if mode == tf.estimator.ModeKeys.TRAIN:
    file_pattern = hparams.train_file
    shuffle = True
  elif mode == tf.estimator.ModeKeys.EVAL:
    file_pattern = hparams.eval_file
    shuffle = False
  else:
    raise NotImplementedError('Unsupported mode {}'.format(mode))

  return _get_dataset(
      file_pattern,
      get_input_shape(hparams),
      get_label_shape(hparams),
      get_tfrecord_shape(hparams),
      hparams.batch_size,
      mode=mode,
      compression_type=hparams.compression_type,
      shuffle=shuffle,
      shuffle_buffer_size=hparams.shuffle_buffer_size)

def get_continuous_data(hparams):
  filename = hparams.continuous_file
  continuous_data_list = []
  width, channels = get_input_shape(hparams)
  sps = 20
  cont_width = int(sps * (3600 * 24 + 20))
  cont_shape = (cont_width, channels)
  window_size = 400
  shift = window_size // 2

  def sub_to_batch(sub):
    return sub.batch(window_size, drop_remainder=True)

  dataset = tf.data.TFRecordDataset(
      filename, compression_type=hparams.compression_type)
  dataset = dataset.map(
      lambda x: _parse_function(
          x, tf.estimator.ModeKeys.PREDICT, (cont_width, channels), (1,), 0))
  dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
  dataset = dataset.window(window_size, shift=shift)
  dataset = dataset.flat_map(sub_to_batch)
  dataset = dataset.map(lambda x: tf.reshape(x, (1, window_size, channels)))
  return dataset