import tensorflow as tf
from tensorflow.keras import layers

class GroupNorm3D(layers.Layer):
    def __init__(self, G=32, eps=1e-5, **kwargs):
        super(GroupNorm3D, self).__init__(**kwargs)
        self.G = G
        self.eps = eps

    def build(self, input_shape):
        # Get the number of channels (C) from the input shape
        _, D, H, W, C = input_shape.as_list()
        self.G = min(self.G, C)

        # Define trainable parameters gamma and beta
        self.gamma = self.add_weight('gamma', shape=[1, 1, 1, 1, C], initializer='ones', trainable=True)
        self.beta = self.add_weight('beta', shape=[1, 1, 1, 1, C], initializer='zeros', trainable=True)

    def call(self, inputs):
        N, D, H, W, C = inputs.get_shape().as_list()
        G = min(self.G, C)

        # Reshape the input tensor to group the channels
        x = tf.reshape(inputs, [N, D, H, W, G, C // G])

        # Compute the mean and variance along the group dimensions (depth, height, width, group)
        mean, var = tf.nn.moments(x, [1, 2, 3, 5], keepdims=True)

        # Normalize the tensor
        x = (x - mean) / tf.sqrt(var + self.eps)

        # Reshape back to original
        x = tf.reshape(x, [N, D, H, W, C])

        # Apply gamma and beta
        x = x * self.gamma + self.beta

        return x