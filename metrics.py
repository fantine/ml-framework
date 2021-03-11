from typing import Sequence

import tensorflow as tf


class Accuracy(tf.keras.metrics.Accuracy):

  def __init__(self, from_logits: bool = False, **kwargs):
    super(Accuracy, self).__init__(**kwargs)
    self.from_logits = from_logits

  def update_state(self, y_true: Sequence[float], y_pred: Sequence[float]):
    if self.from_logits:
      y_pred = tf.math.sigmoid(y_pred)
    super(Accuracy, self).update_state(y_true, y_pred)


class AUC(tf.keras.metrics.AUC):

  def __init__(self, from_logits: bool = False, **kwargs):
    super(AUC, self).__init__(**kwargs)
    self.from_logits = from_logits

  def update_state(self, y_true: Sequence[float], y_pred: Sequence[float]):
    if self.from_logits:
      y_pred = tf.math.sigmoid(y_pred)
    super(AUC, self).update_state(y_true, y_pred)


class Precision(tf.keras.metrics.Precision):

  def __init__(self, from_logits: bool = False, **kwargs):
    super(Precision, self).__init__(**kwargs)
    self.from_logits = from_logits

  def update_state(self, y_true: Sequence[float], y_pred: Sequence[float]):
    if self.from_logits:
      y_pred = tf.math.sigmoid(y_pred)
    super(Precision, self).update_state(y_true, y_pred)


class Recall(tf.keras.metrics.Recall):

  def __init__(self, from_logits: bool = False, **kwargs):
    super(Recall, self).__init__(**kwargs)
    self.from_logits = from_logits

  def update_state(self, y_true: Sequence[float], y_pred: Sequence[float]):
    if self.from_logits:
      y_pred = tf.math.sigmoid(y_pred)
    super(Recall, self).update_state(y_true, y_pred)

