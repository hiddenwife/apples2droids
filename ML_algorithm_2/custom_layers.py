import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom")
class AbsDiff(layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return tf.abs(a - b)

@register_keras_serializable(package="Custom")
class ReduceMin(layers.Layer):
    def call(self, x):
        return tf.reduce_min(x, axis=1)

@register_keras_serializable(package="Custom")
class ReduceMean(layers.Layer):
    def call(self, x):
        return tf.reduce_mean(x, axis=1)
