import os
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.python import debug as tfdbg
import data_helpers_2 as data_helpers
import time

# State which model to use here
from vdcnn import VDCNN

# Parameters settings
# Data loading params
# tf.flags.DEFINE_string("database_path", "ag_news_csv/", "Path for the dataset to be used.")
tf.flags.DEFINE_string("pos_dir", "data/rt-polaritydata/rt-polarity.pos", "Path of positive data")
tf.flags.DEFINE_string("neg_dir", "data/rt-polaritydata/rt-polarity.neg", "Path of negative data")
tf.flags.DEFINE_float("dev_sample_percentage", 0.001, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_max_length", 50, "Sequence Max Length (default: 1024)")
tf.flags.DEFINE_string("downsampling_type", "maxpool", "Types of downsampling methods, use either three of maxpool, k-maxpool and linear (default: 'maxpool')")
tf.flags.DEFINE_integer("depth", 9, "Depth for VDCNN, use either 9, 17, 29 or 47 (default: 9)")
tf.flags.DEFINE_boolean("use_he_uniform", True, "Initialize embedding lookup with he_uniform (default: True)")
tf.flags.DEFINE_boolean("optional_shortcut", False, "Use optional shortcut (default: False)")
tf.flags.DEFINE_integer("min_frequency", 10, "Min word frequency to be contained in vocab list")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 1e-2, "Starter Learning Rate (default: 1e-2)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_boolean("enable_tensorboard", True, "Enable Tensorboard (default: True)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr, value))
print("")

def train():
	# Data Preparation
	# Load data
	print("Loading data...")
	# data_helper = data_helper(sequence_max_length=FLAGS.sequence_max_length)
	# train_data, train_label, test_data, test_label = data_helper.load_dataset(FLAGS.database_path)

	with tf.device('/cpu:0'):
		x_text, y = data_helpers.load_data_and_labels(FLAGS.pos_dir, FLAGS.neg_dir)

	text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.sequence_max_length,
																			  min_frequency=FLAGS.min_frequency)
	x = np.array(list(text_vocab_processor.fit_transform(x_text)))
	print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))

	print("x = {0}".format(x.shape))
	print("y = {0}".format(y.shape))
	print("")

	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(y)))
	x_shuffled = x[shuffle_indices]
	y_shuffled = y[shuffle_indices]

	# Split train/test set
	# TODO: This is very crude, should use cross-validation
	dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
	x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
	y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
	print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

	num_batches_per_epoch = int((len(x_train)-1)/FLAGS.batch_size) + 1
	print("Loading data succees...")

	# ConvNet
	acc_list = [0]
	sess = tf.Session()

	cnn = VDCNN(num_classes=y_train.shape[1],
		num_quantized_chars=len(text_vocab_processor.vocabulary_),
		depth=FLAGS.depth,
		sequence_max_length=FLAGS.sequence_max_length,
		downsampling_type=FLAGS.downsampling_type,
		use_he_uniform=FLAGS.use_he_uniform,
		optional_shortcut=FLAGS.optional_shortcut)

	# Optimizer and LR Decay
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		global_step = tf.Variable(0, name="global_step", trainable=False)
		learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.num_epochs*num_batches_per_epoch, 0.95, staircase=True)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		gradients, variables = zip(*optimizer.compute_gradients(cnn.loss))
		gradients, _ = tf.clip_by_global_norm(gradients, 7.0)
		train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

	###
	# Output directory for models and summaries
	timestamp = str(int(time.time()))
	out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
	print("Writing to {}\n".format(out_dir))

	# Summaries for loss and accuracy
	loss_summary = tf.summary.scalar("loss", cnn.loss)
	acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

	# Train Summaries
	train_summary_op = tf.summary.merge([loss_summary, acc_summary])
	train_summary_dir = os.path.join(out_dir, "summaries", "train")
	train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

	# Dev summaries
	dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
	dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
	dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

	# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
	checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
	checkpoint_prefix = os.path.join(checkpoint_dir, "model")
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

	# Write vocabulary
	text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))
	###


	# Initialize Graph
	sess.run(tf.global_variables_initializer())

	# sess = tfdbg.LocalCLIDebugWrapperSession(sess)  # 被调试器封装的会话
	# sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)  # 调试器添加过滤规则

	# Train Step and Test Step
	def train_step(x_batch, y_batch):
		"""
		A single training step
		"""
		feed_dict = {cnn.input_x: x_batch,
					 cnn.input_y: y_batch,
					 cnn.is_training: True}
		_, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
		train_summary_writer.add_summary(summaries, step)
		time_str = datetime.datetime.now().isoformat()
		print("{}: Step {}, Epoch {}, Loss {:g}, Acc {:g}".format(time_str, step, int(step//num_batches_per_epoch)+1, loss, accuracy))
		#if step%FLAGS.evaluate_every == 0 and FLAGS.enable_tensorboard:
		#	summaries = sess.run(train_summary_op, feed_dict)
		#	train_summary_writer.add_summary(summaries, global_step=step)

	def test_step(x_batch, y_batch):
		"""
		Evaluates model on a dev set
		"""
		feed_dict = {cnn.input_x: x_batch,
					 cnn.input_y: y_batch,
					 cnn.is_training: False}
		summaries_dev, loss, preds, step = sess.run([dev_summary_op, cnn.loss, cnn.predictions, global_step], feed_dict)
		dev_summary_writer.add_summary(summaries_dev, step)
		time_str = datetime.datetime.now().isoformat()
		return preds, loss

	# Generate batches
	# train_batches = data_helper.batch_iter(list(zip(train_data, train_label)), FLAGS.batch_size, FLAGS.num_epochs)

	batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

	# Training loop. For each batch...
	for train_batch in batches:
		x_batch, y_batch = zip(*train_batch)
		train_step(x_batch, y_batch)
		current_step = tf.train.global_step(sess, global_step)
		# Testing loop
		if current_step % FLAGS.evaluate_every == 0:
			print("\nEvaluation:")
			i = 0
			index = 0
			sum_loss = 0
			test_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1, shuffle=False)
			y_preds = np.ones(shape=len(y_dev), dtype=np.int)
			for test_batch in test_batches:
				x_test_batch, y_test_batch = zip(*test_batch)
				preds, test_loss = test_step(x_test_batch, y_test_batch)
				sum_loss += test_loss
				res = np.absolute(preds - np.argmax(y_test_batch, axis=1))
				y_preds[index:index+len(res)] = res
				i += 1
				index += len(res)

			time_str = datetime.datetime.now().isoformat()
			acc = np.count_nonzero(y_preds==0)/len(y_preds)
			acc_list.append(acc)
			print("{}: Evaluation Summary, Loss {:g}, Acc {:g}".format(time_str, sum_loss/i, acc))
			print("{}: Current Max Acc {:g} in Iteration {}".format(time_str, max(acc_list), int(acc_list.index(max(acc_list))*FLAGS.evaluate_every)))

		if current_step % FLAGS.checkpoint_every == 0:
			path = saver.save(sess, checkpoint_prefix, global_step=current_step)
			print("Saved model checkpoint to {}\n".format(path))

def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()




