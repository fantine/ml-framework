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
