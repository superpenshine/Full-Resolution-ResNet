import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from tqdm import trange
from parse import load_data


class FRRN():

	def __init__(self, x_shp, model='A'):

		self.f = open('debug_log', 'w')
		self.Training = True
		self.x_shp = x_shp
		self.num_class = 12 # we will have an additional label for negative
		self.reg_lambda = 1e-4
		self.learning_rate = 1e-4
		self.train_batch_size = 1
		# if valid batch size too big, GPU boom
		self.valid_batch_size = 1
		self.validation_frequency = 100
		# top K pixels loss
		self.K = 512 * 16
		self.max_iter = 5000

		# build the network
		self._build_placeholder()
		self._build_preprocessing()
		model == 'A' ? self._build_model_a() : self._build_model_b()
		self._build_loss()
		self._build_optim()
		self._build_eval()

	def _build_placeholder(self):
		# Build placeholders.

		# N * H * W * C
		x_in_shp = (self.train_batch_size, self.x_shp[1], self.x_shp[2], self.x_shp[3])

		# Create Placeholders for inputs
		self.x_in = tf.placeholder(tf.float32, shape=x_in_shp)
		self.y_in = tf.placeholder(tf.int64, shape=x_in_shp[:3])

	def _build_preprocessing(self):
		"""Build preprocessing related graph."""

		with tf.variable_scope("Normalization", reuse=tf.AUTO_REUSE):
			# TODO: we will make `n_mean`, `n_range`, `n_mean_in` and
			# `n_range_in` as scalar this time! This is how we often use in
			# CNNs, as we KNOW that these are image pixels, and all pixels
			# should be treated equally!

			# Create placeholders for saving mean, range to a TF variable for
			# easy save/load. Create these variables as well.
			self.n_mean_in = tf.placeholder(tf.float32, shape=())
			self.n_range_in = tf.placeholder(tf.float32, shape=())
			# Make the normalization as a TensorFlow variable. This is to make
			# sure we save it in the graph
			self.n_mean = tf.get_variable(
			    "n_mean", shape=(), trainable=False)
			self.n_range = tf.get_variable(
				"n_range", shape=(), trainable=False)
			# Assign op to store this value to TF variable
			self.n_assign_op = tf.group(
				tf.assign(self.n_mean, self.n_mean_in),
				tf.assign(self.n_range, self.n_range_in),
			)

	# --z------residual stream->---------z_prime--->
	# 	|							^
	# 	|							|
	# 	V							|
	# --y------pooling stream-->---------y_prime--->

	def frr_unit(self, residual_stream, pooling_stream, max_pooling_pool_size, max_pooling_strides,
					 max_pooling_padding, conv2d_num_filter, conv2d_kernel, conv2d_strides, conv2d_padding, activ, training=True):
		# iter1: pooling_stream[?, 360, 480, 48], residual stream[?, 720, 960, 32]-->[?, 360, 480, 32] by maxpooling
		residual_stream_orig = residual_stream
		residual_stream = tf.layers.max_pooling2d(residual_stream, max_pooling_pool_size, max_pooling_strides, padding=max_pooling_padding)
		#concatenate 2 stream for conv2d
		# iter1: pooling_stream[?, 360, 480, 80]
		pooling_stream = tf.concat([pooling_stream, residual_stream], axis = 3)
		# iter1: pooling_stream[?, 358, 478, 96]
		pooling_stream = activ(tf.layers.conv2d(pooling_stream, conv2d_num_filter, conv2d_kernel, strides=conv2d_strides, padding=conv2d_padding))
		pooling_stream = tf.layers.batch_normalization(pooling_stream, axis=-1, momentum=1.0, center=True, scale=True, training=training,
														 trainable=True, reuse=None)
		# iter1: pooling_stream[?, 356, 476, 96]
		pooling_stream = activ(tf.layers.conv2d(pooling_stream, conv2d_num_filter, conv2d_kernel, strides=conv2d_strides, padding=conv2d_padding))
		pooling_stream = tf.layers.batch_normalization(pooling_stream, axis=-1, momentum=1.0, center=True, scale=True, training=training,
														 trainable=True, reuse=None)
		y_prime = tf.nn.relu(pooling_stream)
		# this one have same kernel and strids
		# iter1: pooling_stream[?, 356, 476, 32]
		pooling_stream = activ(tf.layers.conv2d(y_prime, 32, (1, 1), strides=(1, 1), padding='same'))
		# iter1: pooling_stream[?, 712, 952, 32]
		pooling_stream = tf.image.resize_nearest_neighbor(pooling_stream, tf.shape(y_prime)[1:3] * max_pooling_strides)
		# concatenate
		z_prime = residual_stream_orig + pooling_stream

		return z_prime, y_prime

	def residual_unit(self, fan_in, num_filter, kernel, strides, padding, activ, traning=True):

		cur_in = activ(tf.layers.conv2d(fan_in, num_filter, kernel, strides=strides, padding=padding))
		cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=traning,
												 trainable=True, reuse=None)
		cur_in = tf.nn.relu(cur_in)
		cur_in = activ(tf.layers.conv2d(cur_in, num_filter, kernel, strides=strides, padding=padding))
		cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=traning,
												 trainable=True, reuse=None)
		
		return cur_in + fan_in

	def _build_model_a(self):

		activ = tf.nn.relu
		kernel_initializer = tf.keras.initializers.he_normal()
		
		with tf.variable_scope("FRRN_A", reuse = tf.AUTO_REUSE):
			# batch_norm
			cur_in = (self.x_in - self.n_mean) / self.n_range
			# before 3 * ResNet
			cur_in = tf.layers.conv2d(cur_in, 48, (5, 5), strides=(1, 1), padding='same', name='conv2d_1')
			cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=self.Training,
													 trainable=True, name="conv2d_1_bn", reuse=None)
			cur_in = activ(cur_in)
			# 3 * RU
			for i in range(3):
				cur_in = self.residual_unit(cur_in, 48, (3, 3), (1, 1), 'same', activ, traning=True)

			# now the shape should be as [?, 720, 960, 48]->[?, 720, 960, 32]. Splited to cur_in & residual_stream
			residual_stream = activ(tf.layers.conv2d(cur_in, 32, (1, 1), strides=(1, 1), padding='same', name='conv2d_to_split'))

			# encoding
			pooling_stream = cur_in
			for it, num_filter, scale in [(3, 96, 2), (4, 192, 4), (2, 384, 8), (2, 384, 16)]:
				# max pool the pooling stream only
				# iter1: [?, 720, 960, 48]->[?, 360, 480, 48]
				pooling_stream = tf.layers.max_pooling2d(pooling_stream, (2, 2), (2, 2), padding='same')

				for i in range(it):
					residual_stream, pooling_stream = self.frr_unit(residual_stream, pooling_stream, scale, scale, 'same',
																	 num_filter, (3, 3), (1, 1), 'same', activ, training=self.Training)

			# decoding
			for it, num_filter, scale in [(2, 192, 8),(2, 192, 4),(2, 96, 2)]:
				# invert the max pool operation
				pooling_stream = tf.image.resize_nearest_neighbor(pooling_stream, tf.shape(pooling_stream)[1:3] * 2)
				# in the very last iteration, we discard pooing_stream actually, back to residual only
				for i in range(it):
					residual_stream, pooling_stream = self.frr_unit(residual_stream, pooling_stream, scale, scale, 'same',
																	 num_filter, (3, 3), (1, 1), 'same', activ, training=self.Training)

			# encoding pooled 4 times, 3 times in decoding, so do the last unpooling
			residual_stream = tf.image.resize_nearest_neighbor(residual_stream, tf.shape(pooling_stream)[1:3] * 2)
			# Concat Streams, make the residual stream back to 48 channels, just rename it back to cur_in
			cur_in = activ(tf.layers.conv2d(residual_stream, 48, (1, 1), strides=(1, 1), padding='same', name='conv2d_to_merge'))
			# 3 * RU again
			for i in range(3):
				cur_in = self.residual_unit(cur_in, 48, (3, 3), (1, 1), 'same', activ, traning=True)
			
			# Final classification layer
			self.logits = activ(tf.layers.conv2d(cur_in, self.num_class + 1, (1, 1), strides=(1, 1), padding='same', name='output_layer'))
			
			# Get list of all weights in this scope. They are called "kernel" in tf.layers.dense.
			self.kernels_list = [_v for _v in tf.trainable_variables() if "kernel" in _v.name]

	def _build_model_b(self):

		activ = tf.nn.relu
		kernel_initializer = tf.keras.initializers.he_normal()
		
		with tf.variable_scope("FRRN_B", reuse = tf.AUTO_REUSE):
			# batch_norm
			cur_in = (self.x_in - self.n_mean) / self.n_range
			# before 3 * ResNet
			cur_in = tf.layers.conv2d(cur_in, 48, (5, 5), strides=(1, 1), padding='same', name='conv2d_1')
			cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True, scale=True, training=self.Training,
													 trainable=True, name="conv2d_1_bn", reuse=None)
			cur_in = activ(cur_in)
			# 3 * RU
			for i in range(3):
				cur_in = self.residual_unit(cur_in, 48, (3, 3), (1, 1), 'same', activ, traning=True)

			# now the shape should be as [?, 720, 960, 48]->[?, 720, 960, 32]. Splited to cur_in & residual_stream
			residual_stream = activ(tf.layers.conv2d(cur_in, 32, (1, 1), strides=(1, 1), padding='same', name='conv2d_to_split'))

			# encoding
			pooling_stream = cur_in
			for it, num_filter, scale in [(3, 96, 2), (4, 192, 4), (2, 384, 8), (2, 384, 16), (2, 384, 32)]:
				# max pool the pooling stream only
				# iter1: [?, 720, 960, 48]->[?, 360, 480, 48]
				pooling_stream = tf.layers.max_pooling2d(pooling_stream, (2, 2), (2, 2), padding='same')

				for i in range(it):
					residual_stream, pooling_stream = self.frr_unit(residual_stream, pooling_stream, scale, scale, 'same',
																	 num_filter, (3, 3), (1, 1), 'same', activ, training=self.Training)

			# decoding
			for it, num_filter, scale in [(2, 192, 16),(2, 192, 8),(2, 192, 4),(2, 96, 2)]:
				# invert the max pool operation
				pooling_stream = tf.image.resize_nearest_neighbor(pooling_stream, tf.shape(pooling_stream)[1:3] * 2)
				# in the very last iteration, we discard pooing_stream actually, back to residual only
				for i in range(it):
					residual_stream, pooling_stream = self.frr_unit(residual_stream, pooling_stream, scale, scale, 'same',
																	 num_filter, (3, 3), (1, 1), 'same', activ, training=self.Training)

			# encoding pooled 4 times, 3 times in decoding, so do the last unpooling
			residual_stream = tf.image.resize_nearest_neighbor(residual_stream, tf.shape(pooling_stream)[1:3] * 2)
			# Concat Streams, make the residual stream back to 48 channels, just rename it back to cur_in
			cur_in = activ(tf.layers.conv2d(residual_stream, 48, (1, 1), strides=(1, 1), padding='same', name='conv2d_to_merge'))
			# 3 * RU again
			for i in range(3):
				cur_in = self.residual_unit(cur_in, 48, (3, 3), (1, 1), 'same', activ, traning=True)
			
			# Final classification layer
			self.logits = activ(tf.layers.conv2d(cur_in, self.num_class + 1, (1, 1), strides=(1, 1), padding='same', name='output_layer'))
			
			# Get list of all weights in this scope. They are called "kernel" in tf.layers.dense.
			self.kernels_list = [_v for _v in tf.trainable_variables() if "kernel" in _v.name]

	def _build_loss(self):
		"""Build our cross entropy loss."""

		with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):
			# Create cross entropy loss, shape of naive_loss is N * H * W * 1
			self.logits = tf.check_numerics(self.logits, "logits contains Nan !!!!!")
			self.naive_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_in, logits=self.logits)
			self.naive_loss = tf.check_numerics(self.naive_loss, "naive_loss contains Nan !!!!!")
			self.loss_shape = (tf.shape(self.logits)[0], tf.shape(self.logits)[1] * tf.shape(self.logits)[2])
			# boot strapped corss-entropy loss
			self.loss_reshaped, _ = tf.nn.top_k(tf.reshape(self.naive_loss, self.loss_shape), self.K, sorted=False)
			self.loss = tf.reduce_mean(self.naive_loss)
			# self.loss = tf.Print(self.loss, [self.loss], "loss")

			# Create l2 regularizer loss and add
			l2_loss = tf.add_n([tf.reduce_sum(_v**2) for _v in self.kernels_list])
			self.loss += self.reg_lambda * l2_loss

			# Record summary for loss
			tf.summary.scalar("loss", self.loss)

	def _build_optim(self):
		"""Build optimizer related ops and vars."""

		with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
			self.global_step = tf.get_variable("global_step", shape=(), initializer=tf.zeros_initializer(), dtype=tf.int64, trainable=False)
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			self.optim = self.optimizer.minimize(self.loss, global_step=self.global_step)

	def _build_eval(self):
		"""Build the evaluation related ops"""

		with tf.variable_scope("Eval", tf.AUTO_REUSE):

			# Compute the accuracy of the model. When comparing labels
			# elemwise, use tf.equal instead of `==`. `==` will evaluate if
			# your Ops are identical Ops.
			self.pred = tf.argmax(self.logits, axis=3)
			self.acc = tf.reduce_mean(
				tf.to_float(tf.equal(self.pred, self.y_in))
			)

			# Record summary for accuracy
			tf.summary.scalar("accuracy", self.acc)

	def train(self, x_tr, y_tr, x_va, y_va):

		x_tr_mean = x_tr.mean()
		x_tr_range = 128.0

		# ----------------------------------------
		# Run TensorFlow Session
		with tf.Session() as sess:
			# Init
			print("Initializing...")
			sess.run(tf.global_variables_initializer())

			# Assign normalization variables from statistics of the train data
			sess.run(self.n_assign_op, feed_dict={
				self.n_mean_in: x_tr_mean,
				self.n_range_in: x_tr_range,
			})

			step = 0
			best_acc = 0
			print("Training...")

			# for each epoch
			for step in trange(step, self.max_iter):

					# Get a random training batch. Notice that we are now going to
					# forget about the `epoch` thing. Theoretically, they should do
					# almost the same.
					ind_cur = np.random.choice(
						len(x_tr), self.train_batch_size, replace=False)
					x_b = np.array([x_tr[_i] for _i in ind_cur])
					y_b = np.array([y_tr[_i] for _i in ind_cur])
					y_b[y_b < 0] = self.num_class
					# Run the operations necessary for training
					res = sess.run(
						fetches={"optim": self.optim,
								"loss": self.loss,
								"logits": self.logits,
						},
						feed_dict={
							self.x_in: x_b,
							self.y_in: y_b,
						},
					)
					print("iteration {} training loss: {}".format(step, res['loss']))
					self.f.write("max: "+str(np.max(res['logits'])))
					self.f.write("min: "+str(np.min(res['logits'])))

					# TODO: Validate every N iterations and at the first
					# iteration. Use `self.config.val_freq`. Make sure that we
					# validate at the correct iterations. HINT: should be similar
					# to above.
					b_validate = ((step + 1) % self.validation_frequency == 0) or (step == 0)
					if b_validate:
						num_valid_b = len(x_va) // self.valid_batch_size
						acc_list = []
						for idx_b in range(num_valid_b):
							res = sess.run(
								fetches={
									"acc": self.acc,
									"global_step": self.global_step,
								},
								feed_dict={
									self.x_in: x_va[idx_b * self.valid_batch_size: (idx_b + 1) * self.valid_batch_size],
									self.y_in: y_va[idx_b * self.valid_batch_size: (idx_b + 1) * self.valid_batch_size],
								}
							)
							acc_list += [res['acc']]
						# If best validation accuracy, update W_best, b_best, and
						# best accuracy. We will only return the best W and b
						avg_acc = np.mean(acc_list)
						if avg_acc > best_acc:
							best_acc = avg_acc

			print("best training validation accuracy is: {}".format(best_acc))
			print("Closing TF Session")

def main():

	print("Reading training data...")
	x_trva, y_trva = load_data("./data.h5", 12)

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
	frrn_a = FRRN('A', (10, 720, 960, 3))

	# ----------------------------------------
	# Train
	# Run training
	frrn_a.train(x_tr, y_tr, x_va, y_va)

if __name__ == "__main__":

    main()
