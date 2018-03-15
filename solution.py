import numpy as np
import tensorflow as tf

from tqdm import trange
from parse import load_data


class FRRN_A():

	def __init__(self, x_shp):

		self.x_shp = x_shp

		# build the network
		self._build_placeholder()
		self._build_model()

	def _build_placeholder(self):
		# Build placeholders.

		# N * H * W * C
		x_in_shp = (None, self.x_shp[1], self.x_shp[2], self.x_shp[3])

		# Create Placeholders for inputs
		self.x_in = tf.placeholder(tf.float32, shape=x_in_shp)
		self.y_in = tf.placeholder(tf.float32, shape=x_in_shp)

	def residual_unit(self, fan_in, num_filter, kernel, strides, activ, padding, traning=True):

		cur_in = tf.layers.conv2d(fan_in, num_filter, kernel, strides=strides, padding=padding)
		cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=traning, trainable=True, reuse=None)
		cur_in = activ(cur_in)
		cur_in = tf.layers.conv2d(cur_in, num_filter, kernel, strides=strides, padding=padding)
		cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=traning, trainable=True, reuse=None)
		
		return cur_in + fan_in

	def _build_model(self):

		activ = tf.nn.relu
		kernel_initializer = tf.keras.initializers.he_normal()
		
		with tf.variable_scope("FRRN_A", reuse = tf.AUTO_REUSE):
			# before 3 * ResNet
			cur_in = tf.layers.conv2d(self.x_in, 48, (5, 5), strides=(1, 1), padding='same', name='conv2d_1')
			cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=True, trainable=True, name="conv2d_1_bn", reuse=None)
			cur_in = activ(cur_in)

			for i in range(3):
				cur_in = self.residual_unit(cur_in, 48, (3, 3), (1, 1), activ, 'same', traning=True)

def main():

	frrn_a = FRRN_A((10, 720, 960, 3))


if __name__ == "__main__":

    main()
