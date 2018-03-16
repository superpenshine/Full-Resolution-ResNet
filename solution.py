import numpy as np
import tensorflow as tf

from tqdm import trange
from parse import load_data


class FRRN_A():

	def __init__(self, x_shp):

		self.x_shp = x_shp
		self.num_class = 10

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

# --z------residual stream->---------z_prime--->
# 	|							^
# 	|							|
# 	|							|
# --y-|---pooling stream--->---------y_prime--->

	def frr_unit(self, residual_stream, pooling_stream, max_pooling_pool_size, max_pooling_strides,
					 max_pooling_padding, conv2d_num_filter, conv2d_kernel, conv2d_strides, conv2d_padding, training=True):

		pooling_stream = tf.layers.max_pooling2d(pooling_stream, max_pooling_pool_size, max_pooling_strides, padding=max_pooling_padding)
		pooling_stream = tf.layers.conv2d(pooling_stream, conv2d_num_filter, conv2d_kernel, strides=conv2d_strides, padding=conv2d_padding)
		pooling_stream = tf.layers.batch_normalization(pooling_stream, axis=-1, momentum=1.0, center=True, scale=True, training=training,
														 trainable=True, reuse=None)
		pooling_stream = tf.layers.conv2d(pooling_stream, conv2d_num_filter, conv2d_kernel, strides=conv2d_strides, padding=conv2d_padding)
		pooling_stream = tf.layers.batch_normalization(pooling_stream, axis=-1, momentum=1.0, center=True, scale=True, training=training,
														 trainable=True, reuse=None)
		y_prime = tf.nn.relu(pooling_stream)
		# this one have same kernel and strids
		pooling_stream = tf.layers.conv2d(y_prime, 32, (1, 1), strides=(1, 1), padding='same')
		pooling_stream = tf.image.resize_nearest_neighbor(pooling_stream, tf.shape(y_prime)[1:3] * max_pooling_strides)
		# concatenate
		z_prime = residual_stream + pooling_stream

		return z_prime, y_prime

	def residual_unit(self, fan_in, num_filter, kernel, strides, padding, traning=True):

		cur_in = tf.layers.conv2d(fan_in, num_filter, kernel, strides=strides, padding=padding)
		cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=traning,
												 trainable=True, reuse=None)
		cur_in = tf.nn.relu(cur_in)
		cur_in = tf.layers.conv2d(cur_in, num_filter, kernel, strides=strides, padding=padding)
		cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=traning,
												 trainable=True, reuse=None)
		
		return cur_in + fan_in

	def _build_model(self):

		activ = tf.nn.relu
		kernel_initializer = tf.keras.initializers.he_normal()
		Training = True
		
		with tf.variable_scope("FRRN_A", reuse = tf.AUTO_REUSE):
			# before 3 * ResNet
			cur_in = tf.layers.conv2d(self.x_in, 48, (5, 5), strides=(1, 1), padding='same', name='conv2d_1')
			cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=Training,
													 trainable=True, name="conv2d_1_bn", reuse=None)
			cur_in = activ(cur_in)
			# 3 * RU
			for i in range(3):
				cur_in = self.residual_unit(cur_in, 48, (3, 3), (1, 1), 'same', traning=True)

			# now the shape should be as [?, 720, 960, 48], make it [?, 720, 960, 32]. Splited to cur_in & residual_stream
			residual_stream = tf.layers.conv2d(cur_in, 32, (1, 1), strides=(1, 1), padding='same', name='conv2d_to_split')
			
			# encoding
			for it, num_filter, scale in [(3, 96, 2), (4, 192, 4), (2, 384, 8), (2, 384, 16)]:
				# max pool the pooling stream only
				
				pooling_stream = tf.layers.max_pooling2d(cur_in, (2, 2), (2, 3), padding='same')
				for i in range(it):
					residual_stream, pooling_stream = self.frr_unit(residual_stream, pooling_stream, scale, scale, 'same',
																	 num_filter, (3, 3), (1, 1), 'same', training=Training)

			# decoding
			for it, num_filter, scale in [(2, 192, 8),(2, 192, 4),(2, 96, 2)]:
				# invert the max pool operation
				pooling_stream = tf.image.resize_nearest_neighbor(pooling_stream, tf.shape(pooling_stream)[1:3] * 2)
				# in the very last iteration, we discard pooing_stream actually, back to residual only
				for i in range(it):
					residual_stream, pooling_stream = self.frr_unit(residual_stream, pooling_stream, scale, scale, 'same',
																	 num_filter, (3, 3), (1, 1), 'same', training=Training)

			# encoding pooled 4 times, 3 times in decoding, so do the last unpooling
			residual_stream = tf.image.resize_nearest_neighbor(residual_stream, tf.shape(pooling_stream)[1:3] * 2)
			# Concat Streams, make the residual stream back to 48 channels, just call it back to cur_in
			cur_in = tf.layers.conv2d(residual_stream, 48, (1, 1), strides=(1, 1), padding='same', name='conv2d_to_merge')
			# 3 * RU again
			for i in range(3):
				cur_in = self.residual_unit(cur_in, 48, (3, 3), (1, 1), 'same', traning=True)
			
			# Final classification layer
			self.logits = tf.layers.conv2d(cur_in, self.num_class, (1, 1), strides=(1, 1), padding='same', name='output_layer')	
			# Get list of all weights in this scope. They are called "kernel" in tf.layers.dense.
			self.kernels_list = [_v for _v in tf.trainable_variables() if "kernel" in _v.name]

    def _build_loss(self):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):

            # Create cross entropy loss
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_in, logits=self.logits)
            )

            # Create l2 regularizer loss and add
            l2_loss = tf.add_n([
                tf.reduce_sum(_v**2) for _v in self.kernels_list])
            self.loss += self.config.reg_lambda * l2_loss

            # Record summary for loss
            tf.summary.scalar("loss", self.loss)

def main():

	frrn_a = FRRN_A((10, 720, 960, 3))


if __name__ == "__main__":

    main()
