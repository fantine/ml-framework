from tensorflow import keras


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


def residual_block_1d(inputs, filters, strides, regularizer, batchnorm):
  """Creates 1D convolution layer block with residual connections.

  Args:
    inputs: Input to the residual block.
    filters: List of filters to use in successive layers.
    strides: List of strides to use in successive layers.
    regularizer: Regularizer.
    batchnorm: Whether or not to use batch normalization.

  Returns:
    Output of the residual block.
  """
  if batchnorm == 1:
    x = keras.layers.BatchNormalization()(inputs)
  else:
    x = inputs
  x = keras.layers.LeakyReLU()(x)
  if strides[0] == 1:
    x = keras.layers.Conv1D(
        filters=filters[0],
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizer)(x)
  else:
    x = keras.layers.MaxPooling1D(pool_size=2, strides=strides[0])(x)

  if batchnorm == 1:
    x = keras.layers.BatchNormalization()(x)
  x = keras.layers.LeakyReLU()(x)
  x = keras.layers.Conv1D(
      filters=filters[1],
      kernel_size=3,
      strides=strides[1],
      padding='same',
      use_bias=False,
      kernel_regularizer=regularizer)(x)
  # Construct the residual link that bypasses this block.
  shortcut = keras.layers.Conv1D(
      filters=filters[1],
      kernel_size=1,
      strides=strides[0],
      padding='same',
      use_bias=False,
      kernel_regularizer=regularizer)(inputs)
  if batchnorm == 1:
    shortcut = keras.layers.BatchNormalization()(shortcut)

  return shortcut + x


def residual_block_2d(inputs, filters, strides, regularizer, batchnorm):
  """Creates 2D convolution layer block with residual connections.

  Args:
    inputs: Input to the residual block.
    filters: List of filters to use in successive layers.
    strides: List of strides to use in successive layers.
    regularizer: Regularizer.
    batchnorm: Whether or not to use batch normalization.

  Returns:
    Output of the residual block.
  """
  if batchnorm == 1:
    x = keras.layers.BatchNormalization()(inputs)
  else:
    x = inputs
  x = keras.layers.LeakyReLU()(x)
  if strides[0] == 1:
    x = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizer)(x)
  else:
    x = keras.layers.MaxPooling2D(pool_size=2, strides=strides[0])(x)

  if batchnorm == 1:
    x = keras.layers.BatchNormalization()(x)
  x = keras.layers.LeakyReLU()(x)
  x = keras.layers.Conv2D(
      filters=filters[1],
      kernel_size=3,
      strides=strides[1],
      padding='same',
      use_bias=False,
      kernel_regularizer=regularizer)(x)
  # Construct the residual link that bypasses this block.
  shortcut = keras.layers.Conv2D(
      filters=filters[1],
      kernel_size=1,
      strides=strides[0],
      padding='same',
      use_bias=False,
      kernel_regularizer=regularizer)(inputs)
  if batchnorm == 1:
    shortcut = keras.layers.BatchNormalization()(shortcut)

  return shortcut + x
