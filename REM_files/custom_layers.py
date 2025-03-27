import tensorflow as tf
from tensorflow.keras import layers

class GroupNorm(layers.Layer):
    def __init__(self, G=32, eps=1e-5, **kwargs):
        super(GroupNorm, self).__init__(**kwargs)
        self.G = G
        self.eps = eps

    def build(self, input_shape):
        # Get the number of channels (C) from the input shape
        _, _, _, C = input_shape.as_list()
        self.G = min(self.G, C)

        # Define trainable parameters gamma and beta
        self.gamma = self.add_weight('gamma', shape=[1, 1, 1, C], initializer='ones', trainable=True)
        self.beta = self.add_weight('beta', shape=[1, 1, 1, C], initializer='zeros', trainable=True)

    def call(self, inputs):
        N, H, W, C = inputs.get_shape().as_list()
        G = min(self.G, C)

        # Reshape the input tensor
        x = tf.reshape(inputs, [N, H, W, G, C // G])

        # Compute the mean and variance along the group dimensions
        mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)

        # Normalize the tensor
        x = (x - mean) / tf.sqrt(var + self.eps)

        # Reshape back to original
        x = tf.reshape(x, [N, H, W, C])

        # Apply gamma and beta
        x = x * self.gamma + self.beta

        return x