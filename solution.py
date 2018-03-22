import numpy as np
import tensorflow as tf

from tqdm import trange
from parse import load_data


class FRRN_A():

	def __init__(self, x_shp):


		self.Training = True
		self.x_shp = x_shp
		self.num_class = 10
		self.reg_lambda = 1e-4
		self.learning_rate = 1e-3

		# build the network
		self._build_placeholder()
		self._build_model()
		self._build_loss()
		self._build_optim()

	def _build_placeholder(self):
		# Build placeholders.

		# N * H * W * C
		x_in_shp = (None, self.x_shp[1], self.x_shp[2], self.x_shp[3])

		# Create Placeholders for inputs
		self.x_in = tf.placeholder(tf.float32, shape=x_in_shp)
		self.y_in = tf.placeholder(tf.int64, shape=x_in_shp[:3])

# --z------residual stream->---------z_prime--->
# 	|							^
# 	|							|
# 	V							|
# --y------pooling stream-->---------y_prime--->

	def frr_unit(self, residual_stream, pooling_stream, max_pooling_pool_size, max_pooling_strides,
					 max_pooling_padding, conv2d_num_filter, conv2d_kernel, conv2d_strides, conv2d_padding, training=True):
		# iter1: pooling_stream[?, 360, 480, 48], residual stream[?, 720, 960, 32]-->[?, 360, 480, 32] by maxpooling
		# residual_stream = residual_stream+pooling_stream
		residual_stream_orig = residual_stream
		residual_stream = tf.layers.max_pooling2d(residual_stream, max_pooling_pool_size, max_pooling_strides, padding=max_pooling_padding)
		#concatenate 2 stream for conv2d
		# iter1: pooling_stream[?, 360, 480, 80]
		print("pool", pooling_stream)
		print("residual", residual_stream)
		pooling_stream = tf.concat([pooling_stream, residual_stream], axis = 3)
		# iter1: pooling_stream[?, 358, 478, 96]
		pooling_stream = tf.layers.conv2d(pooling_stream, conv2d_num_filter, conv2d_kernel, strides=conv2d_strides, padding=conv2d_padding)
		pooling_stream = tf.layers.batch_normalization(pooling_stream, axis=-1, momentum=1.0, center=True, scale=True, training=training,
														 trainable=True, reuse=None)
		# iter1: pooling_stream[?, 356, 476, 96]
		pooling_stream = tf.layers.conv2d(pooling_stream, conv2d_num_filter, conv2d_kernel, strides=conv2d_strides, padding=conv2d_padding)
		pooling_stream = tf.layers.batch_normalization(pooling_stream, axis=-1, momentum=1.0, center=True, scale=True, training=training,
														 trainable=True, reuse=None)
		y_prime = tf.nn.relu(pooling_stream)
		# this one have same kernel and strids
		# iter1: pooling_stream[?, 356, 476, 32]
		pooling_stream = tf.layers.conv2d(y_prime, 32, (1, 1), strides=(1, 1), padding='same')
		# iter1: pooling_stream[?, 712, 952, 32]
		pooling_stream = tf.image.resize_nearest_neighbor(pooling_stream, tf.shape(y_prime)[1:3] * max_pooling_strides)
		# concatenate
		z_prime = residual_stream_orig + pooling_stream

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
		
		with tf.variable_scope("FRRN_A", reuse = tf.AUTO_REUSE):
			# before 3 * ResNet
			cur_in = tf.layers.conv2d(self.x_in, 48, (5, 5), strides=(1, 1), padding='same', name='conv2d_1')
			cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=self.Training,
													 trainable=True, name="conv2d_1_bn", reuse=None)
			cur_in = activ(cur_in)
			# 3 * RU
			for i in range(3):
				cur_in = self.residual_unit(cur_in, 48, (3, 3), (1, 1), 'same', traning=True)

			# now the shape should be as [?, 720, 960, 48]->[?, 720, 960, 32]. Splited to cur_in & residual_stream
			residual_stream = tf.layers.conv2d(cur_in, 32, (1, 1), strides=(1, 1), padding='same', name='conv2d_to_split')
			
			# encoding
			pooling_stream = cur_in
			for it, num_filter, scale in [(3, 96, 2), (4, 192, 4), (2, 384, 8), (2, 384, 16)]:
				# max pool the pooling stream only
				# iter1: [?, 720, 960, 48]->[?, 360, 480, 48]
				pooling_stream = tf.layers.max_pooling2d(pooling_stream, (2, 2), (2, 2), padding='same')

				for i in range(it):
					residual_stream, pooling_stream = self.frr_unit(residual_stream, pooling_stream, scale, scale, 'same',
																	 num_filter, (3, 3), (1, 1), 'same', training=self.Training)

			# decoding
			for it, num_filter, scale in [(2, 192, 8),(2, 192, 4),(2, 96, 2)]:
				# invert the max pool operation
				pooling_stream = tf.image.resize_nearest_neighbor(pooling_stream, tf.shape(pooling_stream)[1:3] * 2)
				# in the very last iteration, we discard pooing_stream actually, back to residual only
				for i in range(it):
					residual_stream, pooling_stream = self.frr_unit(residual_stream, pooling_stream, scale, scale, 'same',
																	 num_filter, (3, 3), (1, 1), 'same', training=self.Training)

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
			self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_in, logits=self.logits))

			# Create l2 regularizer loss and add
			l2_loss = tf.add_n([tf.reduce_sum(_v**2) for _v in self.kernels_list])
			self.loss += self.reg_lambda * l2_loss

			# Record summary for loss
			tf.summary.scalar("loss", self.loss)

	def _build_optim(self):
		"""Build optimizer related ops and vars."""

		with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
			self.global_step = tf.get_variable("global_step", shape=(), initializer=tf.zeros_initializer(), dtype=tf.int64, trainable=False)
			optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			self.optim = optimizer.minimize(self.loss, global_step=self.global_step)

	def train(self, x_tr, y_tr, x_va, y_va):

		# ----------------------------------------
		# Run TensorFlow Session
		with tf.Session() as sess:
			# Init
			print("Initializing...")
			sess.run(tf.global_variables_initializer())

			step = 0
			best_acc = 0
			print("Training...")
			batch_size = 1
			max_iter = 5

			# for each epoch
			for step in trange(step, max_iter):

					# Get a random training batch. Notice that we are now going to
					# forget about the `epoch` thing. Theoretically, they should do
					# almost the same.
					ind_cur = np.random.choice(
						len(x_tr), batch_size, replace=False)
					x_b = np.array([x_tr[_i] for _i in ind_cur])
					y_b = np.array([y_tr[_i] for _i in ind_cur])
					print("start training with an epoch")
					# Run the operations necessary for training
					res = sess.run(
						fetches={"optim": self.optim,},
						feed_dict={
							self.x_in: x_b,
							self.y_in: y_b,
						},
					)
					print("finished training with an epoch")
					# TODO: Validate every N iterations and at the first
					# iteration. Use `self.config.val_freq`. Make sure that we
					# validate at the correct iterations. HINT: should be similar
					# to above.
					b_validate = (step % 10 == 0) or (step == max_iter - 1)
					if b_validate:
						res = sess.run(
							fetches={
								"acc": self.acc,
								"global_step": self.global_step,
							},
							feed_dict={
								self.x_in: x_va,
								self.y_in: y_va
							})
						# If best validation accuracy, update W_best, b_best, and
						# best accuracy. We will only return the best W and b
						if res["acc"] > best_acc:
							best_acc = res["acc"]

			print("best training validation accuracy is: {}".format(best_acc))
			print("Closing TF Session")

def main():

	print("Reading training data...")
	x_trva, y_trva = load_data("E:\\data\\SYNTHIA\\data.h5", 3)

	# Randomly shuffle data and labels. IMPORANT: make sure the data and label
	# is shuffled with the same random indices so that they don't get mixed up!
	idx_shuffle = np.random.permutation(len(x_trva))
	x_trva = x_trva[idx_shuffle]
	y_trva = y_trva[idx_shuffle]

	# Change type to float32 and int64 since we are going to use that for
	# TensorFlow.
	x_trva = x_trva.astype("float32")
	y_trva = y_trva.astype("int64")

	
	# ----------------------------------------
	# Simply select the last 20% of the training data as validation dataset.
	num_tr = int(len(x_trva) * 0.8)
	x_tr = x_trva[:num_tr]
	x_va = x_trva[num_tr:]
	y_tr = y_trva[:num_tr]
	y_va = y_trva[num_tr:]
	frrn_a = FRRN_A((10, 720, 960, 3))

	# ----------------------------------------
	# Train
	# Run training
	frrn_a.train(x_tr, y_tr, x_va, y_va)

if __name__ == "__main__":

    main()
