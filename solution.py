import os
import numpy as np
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.INFO)
import pdb
from tqdm import trange
from parse import load_data


class FRRN():

    def __init__(self, x_shp, model='A'):

        self.Training = True
        self.x_shp = x_shp
        self.num_class = 12 # we will have an additional label for negative
        self.reg_lambda = 1e-4
        self.learning_rate = 1e-4
        self.train_batch_size = 1
        # if valid batch size too big, GPU boom
        self.valid_batch_size = 1
        self.test_batch_size = 1
        self.validation_frequency = 10

        self.report_freq = 5
        # top K pixels loss
        self.K = 512 * 16
        self.max_iter = 50

        self.log_dir = 'logs'
        self.save_dir = 'save'

        # build the network
        self._build_placeholder()
        self._build_preprocessing()
        self._build_model_a() if model == 'A' else self._build_model_b()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

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
                max_pooling_padding, conv2d_num_filter, conv2d_kernel, conv2d_strides, conv2d_padding, activ,
                training=True):
        # iter1: pooling_stream[?, 360, 480, 48], residual stream[?, 720, 960, 32]-->[?, 360, 480, 32] by maxpooling
        residual_stream_orig = residual_stream
        residual_stream = tf.layers.max_pooling2d(residual_stream, max_pooling_pool_size, max_pooling_strides,
                         padding=max_pooling_padding)
        #concatenate 2 stream for conv2d
        # iter1: pooling_stream[?, 360, 480, 80]
        pooling_stream = tf.concat([pooling_stream, residual_stream], axis = 3)
        # iter1: pooling_stream[?, 358, 478, 96]
        pooling_stream = activ(tf.layers.conv2d(pooling_stream, conv2d_num_filter, conv2d_kernel, strides=conv2d_strides,
                         padding=conv2d_padding))
        pooling_stream = tf.layers.batch_normalization(pooling_stream, axis=-1, momentum=1.0, center=True, scale=True,
                         training=training, trainable=True, reuse=None)
        # iter1: pooling_stream[?, 356, 476, 96]
        pooling_stream = activ(tf.layers.conv2d(pooling_stream, conv2d_num_filter, conv2d_kernel, strides=conv2d_strides,
                         padding=conv2d_padding))
        pooling_stream = tf.layers.batch_normalization(pooling_stream, axis=-1, momentum=1.0, center=True, scale=True,
                         training=training, trainable=True, reuse=None)
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
            cur_in = tf.layers.batch_normalization(cur_in, axis=-1, momentum=1.0, center=True,
                     scale=True, training=self.Training, trainable=True, name="conv2d_1_bn", reuse=None)
            cur_in = activ(cur_in)
            # 3 * RU
            for i in range(3):
                cur_in = self.residual_unit(cur_in, 48, (3, 3), (1, 1), 'same', activ, traning=True)

            # now the shape should be as [?, 720, 960, 48]->[?, 720, 960, 32]. Splited to cur_in & residual_stream
            residual_stream = activ(tf.layers.conv2d(cur_in, 32, (1, 1), strides=(1, 1), padding='same',
                     name='conv2d_to_split'))

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
            self.logits = activ(tf.layers.conv2d(cur_in, self.num_class + 1, (1, 1), strides=(1, 1), padding='same',
                             name='output_layer'))

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
            self.naive_loss = tf.Print(self.naive_loss, [tf.reduce_min(self.naive_loss)], "naive_loss_min")
            self.naive_loss = tf.Print(self.naive_loss, [tf.reduce_max(self.naive_loss)], "naive_loss_max")
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

            # We also want to save best validation accuracy. So we do
            # something similar to what we did before with n_mean. Note that
            # these will also be a scalar variable
            self.best_va_acc_in = tf.placeholder(tf.float32, shape=())
            self.best_va_acc = tf.get_variable("best_va_acc", shape=(), trainable=False)
            self.acc_assign_op = tf.assign(self.best_va_acc, self.best_va_acc_in)

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""

        # Create summary writers (one for train, one for validation)
        self.summary_tr = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
        self.summary_va = tf.summary.FileWriter(os.path.join(self.log_dir, "valid"))
        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.log_dir, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.save_dir, "model")

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

            # Check if previous train exists
            b_resume = tf.train.latest_checkpoint(self.log_dir)
            if b_resume:
                # Restore network
                print("Restoring from {}...".format(
                    self.log_dir))
                self.saver_cur.restore(
                    sess,
                    b_resume
                    )
                # restore number of steps so far
                step = sess.run(self.global_step)
                # restore best acc
                best_acc = sess.run(self.best_va_acc)
            else:
                print("Starting from scratch...")
                step = 0
                best_acc = 0

            print("Training...")
            # for each epoch
            for step in trange(step, self.max_iter):

                self.training=True
                # Get a random training batch. Notice that we are now going to
                # forget about the `epoch` thing. Theoretically, they should do
                # almost the same.
                ind_cur = np.random.choice(
                    len(x_tr), self.train_batch_size, replace=False)
                x_b = np.array([x_tr[_i] for _i in ind_cur])
                y_b = np.array([y_tr[_i] for _i in ind_cur])
                y_b[y_b < 0] = self.num_class

                b_write_summary = ((step + 1) % self.report_freq == 0) or (step == 0)

                if b_write_summary:
                    fetches = {
                        "optim": self.optim,
                        "loss": self.loss,
                        "summary": self.summary_op,
                        "global_step": self.global_step,
                    }
                else:
                    fetches = {
                        "optim": self.optim,
                        "loss": self.loss,
                    }

                # Run the operations necessary for training
                res = sess.run(
                    fetches=fetches,
                    feed_dict={
                        self.x_in: x_b,
                        self.y_in: y_b,
                    },
                )
                print("iteration {} training loss: {}".format(step, res['loss']))
                #
                if 'global_step' in res:
                    print('global_step is: ', res['global_step'])

                    self.summary_tr.add_summary(res["summary"], global_step=res["global_step"],)
                    self.summary_tr.flush()

                    # Also save current model to resume when we write the
                    # summary.
                    self.saver_cur.save(
                        sess, self.save_file_cur,
                        global_step=self.global_step,
                        write_meta_graph=False,
                    )

                b_validate = ((step + 1) % self.validation_frequency == 0) or (step == 0)
                if b_validate:

                    self.traning = False
                    num_valid_b = len(x_va) // self.valid_batch_size
                    acc_list = []
                    loss_list = []

                    for idx_b in range(num_valid_b):
                        res = sess.run(
                            fetches={
                                "acc": self.acc,
                                "loss": self.loss,
                                "global_step": self.global_step,
                            },
                            feed_dict={
                                self.x_in: x_va[idx_b * self.valid_batch_size: (idx_b + 1) * self.valid_batch_size],
                                self.y_in: y_va[idx_b * self.valid_batch_size: (idx_b + 1) * self.valid_batch_size],
                            }
                        )
                        print('valid batch avg loss: ', res['loss'])
                        acc_list += [res['acc']]
                        loss_list += [res['loss']]
                        # pdb.set_trace()
                    # Write Validation Summary over batches
                    avg_loss = np.mean(loss_list)
                    avg_acc = np.mean(acc_list)
                    summary = tf.Summary(value=[
                    	tf.Summary.Value(tag="accuracy", simple_value=avg_acc),
                    	tf.Summary.Value(tag="loss", simple_value=avg_loss),
                    	])
                    self.summary_va.add_summary(summary, global_step=res["global_step"],)
                    self.summary_va.flush()

                    # If best validation accuracy, update W_best, b_best, and
                    # best accuracy. We will only return the best W and b

                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        sess.run(self.acc_assign_op,
                            feed_dict = {
                                self.best_va_acc_in: best_acc,
                            })
                        # Save the best model
                        self.saver_best.save(
                            sess, self.save_file_best,
                            write_meta_graph=False,
                            )

            print("best training validation accuracy is: {}".format(best_acc))
            print("Closing TF Session")

    def test(self, x_te, y_te):
        """Test routine"""

        with tf.Session() as sess:
            # Load the best model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.save_dir)

            if tf.train.latest_checkpoint(self.save_dir) is not None:
                print("Restoring from {}...".format(
                    self.save_dir))
                self.saver_best.restore(
                    sess,
                    latest_checkpoint
                )

            # Test on the test data
            self.training = False
            num_test_b = len(x_te) // self.test_batch_size
            acc_list = []
            for idx_b in range(num_test_b):
                res = sess.run(
                    fetches={
                        "acc": self.acc,
                    },
                    feed_dict={
                        self.x_in: x_va[idx_b * self.test_batch_size: (idx_b + 1) * self.test_batch_size],
                        self.y_in: y_va[idx_b * self.test_batch_size: (idx_b + 1) * self.test_batch_size],
                    }
                )
                acc_list += [res['acc']]

            avg_acc = np.mean(acc_list)

            # Report (print) test result
            print("Test accuracy with the best model is {}".format(res_acc))



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
    frrn_a = FRRN((10, 720, 960, 3), 'A')

    # ----------------------------------------
    # Train
    # Run training
    frrn_a.train(x_tr, y_tr, x_va, y_va)

if __name__ == "__main__":

    main()
