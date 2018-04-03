"""
 " Description: TensorFlow ExpandNet for HDR image reconstruction.
 " Author: LiJinghui
 " Date: March 2018
"""
import tensorflow as tf
import numpy as np
import os, random, time, glob
from datetime import datetime
from math import log10
from model import model
import img_io

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool("is_first_train", "true", "First training or not")
tf.flags.DEFINE_integer("sx", "256", "Image width")
tf.flags.DEFINE_integer("sy", "256", "Image height")
tf.flags.DEFINE_string("data_dir", "../train_data",
                       "Path to processed dataset. This data will be created if the flag [preprocess] is set")
tf.flags.DEFINE_string("output_dir","../train_output", "Path to output directory, for weights and intermediate results")
# Learning parameters
tf.flags.DEFINE_float("num_epochs", "12000.0", "Number of training epochs")
tf.flags.DEFINE_float("start_step", "0.0", "Step to start from")
tf.flags.DEFINE_float("learning_rate", "7e-5", "Starting learning rate for Adam optimizer")
tf.flags.DEFINE_float("update_lr", "0.0007", "Starting learning rate for Adam optimizer")
tf.flags.DEFINE_integer("batch_size", "2", "Batch size for training")
tf.flags.DEFINE_bool("rand_data", "true", "Random shuffling of training data")
tf.flags.DEFINE_bool("print_im","true",  "If images should be printed at each [print_batch_freq] step")

# ===  Create the directory if it doesn't exist ==============================
def dir(path):
    if not os.path.isdir(path):  # Create the log directory if it doesn't exist
        os.makedirs(path)
    return path

# ==============================================================================
data_dir_hdr = dir(os.path.join(FLAGS.data_dir, "hdr"))  # HDR images path
data_dir_sdr = dir(os.path.join(FLAGS.data_dir, "sdr"))  # SDR images path
log_dir = dir(os.path.join(FLAGS.output_dir, "logs"))
im_dir = dir(os.path.join(FLAGS.output_dir, "im"))

# print data_dir_hdr,data_dir_sdr
# Get names of all images in the training path
#frames = [name for name in sorted(os.listdir(data_dir_hdr))]
frames =  sorted(glob.glob(data_dir_hdr + "/*"))
# Randomize the image
if FLAGS.rand_data is not None:
    random.shuffle(frames)
# Get counts of all images in the training path
training_samples = len(frames)
assert training_samples > 0, 'The dataset should not be empty'
print("\n\nData to be used:")
print("\t%d training images" % training_samples)
steps_per_epoch = int(training_samples / FLAGS.batch_size)
max_iters = int(steps_per_epoch * FLAGS.num_epochs)
lr_update_step = 10000


def train():
    with tf.Graph().as_default():
        # === Define training images  ========================================================
        input_data = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.sy, FLAGS.sx, 3])  # (batch_size,256,256,3)
        input_target = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.sy, FLAGS.sx, 3])

        logits = model(input_data)

        global_step = tf.Variable(0, trainable=False)

        # === loss with l2_regularize ========================================================
        # d_l1 = tf.reduce_mean(tf.abs(tf.subtract(logits, input_target)))
        d_l1 = tf.losses.absolute_difference(logits, input_target)
        d_cosin = tf.losses.cosine_distance(tf.nn.l2_normalize(logits, 3), tf.nn.l2_normalize(input_target, 3), dim=3)
        loss = d_l1 + 5 *d_cosin
        tf.add_to_collection('losses', loss)

        regularizer = tf.contrib.layers.l2_regularizer(0.01) #return a l2 regularize function
        reg_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=None) #apply l2_regularizer,return a scale tensor
        tf.add_to_collection('losses', reg_loss)
        loss = tf.add_n(tf.get_collection('losses'))
        #loss = loss + reg_loss
        tf.scalar_summary("cost_function", loss)
        
        # mse score
        #mse = tf.losses.mean_squared_error(logits, input_target)

        # Adam optimizer and learning rate exponential_decay
        start_learning_rate = FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                                   int(lr_update_step), 0.8，staircase=True)

        train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                          epsilon=1e-8, use_locking=False).minimize(loss, global_step = global_step)
        tf.scalar_summary("learning_rate", learning_rate)
        merged_summary_op = tf.merge_all_summaries()

        with tf.Session() as sess:
            # The TensorFlow session to be used
            summary_writer = tf.summary.FileWriter(log_dir, session.graph)
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(tf.global_variables())

            print("\nStarting training...\n")
            if not FLAGS.is_first_train:
                model_file = tf.train.latest_checkpoint('../model_pre')
                saver.restore(sess, model_file)
            tf.logging.set_verbosity(tf.logging.INFO)

            # The training loop
            step = 0
            for epoch in range(int(FLAGS.num_epochs)):
                total_loss = 0.0
                for n_batch in range(int(steps_per_epoch)):
                    x, y_ = load_image_batch(frames, n_batch)
                    feed_dict = {input_data: x, input_target: y_}
                    # err is the loss of one batch
                    _, err, summary_str, y = sess.run([train_op, loss, merged_summary_op, logits], feed_dict=feed_dict)
                    assert not np.isnan(err), 'Model diverged with loss = NaN'
                    #assert not np.isnan(mse), 'Model diverged with loss = NaN'
                    #psnr = 10*log10(255.0*255.0/(mse*65535.0))
                    total_loss += err
                    summary_writer.add_summary(summary_str, step)

                    if step % 100 == 0:
                        # Training statistics
                        print('-------------------------------------------')
                        print('Currently at step %d of %d.' % (step, max_iters))
                        print('%s: loss on training batch is %.5f' % (datetime.now(), total_loss/100.0))
                        print('-------------------------------------------')
                        print('\n')

                    if FLAGS.print_im :
                        # Print samples
                        if step % 1000 == 0:
                            xx = np.squeeze(x[0])
                            yy = np.squeeze(y_[0])
                            y_final = np.squeeze(y[0])
                            img_io.write_LDR(xx, "%s/%06d_in.png" % (im_dir, step))
                            img_io.write_HDR(yy, "%s/%06d_gt.png" % (im_dir, step))
                            img_io.write_HDR(y_final, "%s/%06d_out.png" % (im_dir, step))

                    if step % 5000 == 0 or (step + 1) == max_iters:
                        checkpoint_path = os.path.join('../model', 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

                    step += 1
                print('%s: epoch %d, ave_loss = %.2f' % (datetime.now(), epoch+1, ave_loss))

            print ("\tDone !")


# === Load training data =====================================================
def load_image_batch(frames, b):
    y_batch_path = frames[b * FLAGS.batch_size:(b + 1) * FLAGS.batch_size]
    x_batch_path = [y_img_path.replace("hdr", "sdr") for y_img_path in y_batch_path]
    x, y_ = [], []
    for i in range(int(FLAGS.batch_size)):
        succ, xv, yv = img_io.load_training_pair(x_batch_path[i], y_batch_path[i])
        #xv, yv = img_io.data_aug(xv, yv)
        if not succ:
            continue
        xv = xv[np.newaxis, :, :, :]
        yv = yv[np.newaxis, :, :, :]
        if i == 0:
            x, y_ = xv, yv
        else:
            x = np.concatenate((x, xv), 0)
            y_ = np.concatenate((y_, yv), 0)
    #x = tf.cast(x, tf.float32)
    #y_ = tf.cast(y_, tf.float32)
    return x, y_

train()
