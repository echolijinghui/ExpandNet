"""
 " Description: TensorFlow ExpandNet for HDR image reconstruction.
 " Author: LiJinghui
 " Date: March 2018
"""
import tensorflow as tf
import numpy as np
import os, sys, cv2
import model, img_io

eps = 1e-5

def print_(str, color='', bold=False):
    if color == 'w':
        sys.stdout.write('\033[93m')
    elif color == "e":
        sys.stdout.write('\033[91m')
    elif color == "m":
        sys.stdout.write('\033[95m')

    if bold:
        sys.stdout.write('\033[1m')

    sys.stdout.write(str)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()

# Settings, using TensorFlow arguments
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("width", "1920", "Reconstruction image width")
tf.flags.DEFINE_integer("height", "1048", "Reconstruction image height")
tf.flags.DEFINE_integer("patch_size", "256", "Round to be multiple of 256 to enable global Net")
tf.flags.DEFINE_string("im_dir", "../test_data", "Path to image directory or an individual image")
tf.flags.DEFINE_string("out_dir", "../out", "Path to output directory")
tf.flags.DEFINE_string("params", "hdrcnn_params.npz", "Path to trained CNN weights")

sx = int(np.maximum(256, np.round(FLAGS.width / 256.0) * 256))
sy = int(np.maximum(256, np.round(FLAGS.height / 256.0) * 256))


def print_info():
    print_("\n\n\t-------------------------------------------------------------------\n", 'm')
    print_("\t  HDR image reconstruction from a single exposure using ExpandNet\n\n", 'm')
    print_("\t  Prediction settings...\n", 'm')
    print_("\t  -------------------\n", 'm')
    print_("\t  Input image directory/file:%s\n" % FLAGS.im_dir, 'm')
    print_("\t  Output directory:%s\n" % FLAGS.out_dir, 'm')
    print_("\t  CNN weights:%s\n" % FLAGS.params, 'm')
    print_("\t  Prediction resolution:%dx%d pixels\n" % (sx, sy), 'm')

def forward():
    with tf.Graph().as_default():
        if sx != FLAGS.width or sy != FLAGS.height:
            print_("Warning: ", 'w', True)
            print_("prediction size has been changed from %dx%d pixels to %dx%d\n"%(FLAGS.width, FLAGS.height, sx, sy), 'w')

        # print information
        print_info()

        # If single frame
        frames = [FLAGS.im_dir]

        # If directory is supplied, get names of all files in the path
        if os.path.isdir(FLAGS.im_dir):
            frames = [os.path.join(FLAGS.im_dir, name)
                      for name in sorted(os.listdir(FLAGS.im_dir))]

        # Placeholder for image input and ground truth
        x = tf.placeholder(tf.float32, shape=[1, sy, sx, 3])
        y_ = tf.placeholder(tf.float32, shape=[1, sy, sx, 3])

        # HDR reconstruction autoencoder model
        print_("Network setup:\n")
        y = model.model(x)

        # Load trained CNN weights
        saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)

        print_("\nLoading trained parameters from '%s'..."%FLAGS.params)
        model_file = tf.train.latest_checkpoint(FLAGS.params)
        saver.restore(sess, model_file)
        print_("\tLoad trained parameters done\n")

        if not os.path.exists(FLAGS.out_dir):
            os.makedirs(FLAGS.out_dir)

        print_("\nStarting prediction...\n\n")

        for i in range(len(frames)):
            print("Frame %d: '%s'"%(i,frames[i]))
            try:
                # Read frame
                print_("\tReading...")
                image_sdr_raw, x_buffer_raw = img_io.read_LDR(frames[i])
                print_("\tRead image done")

                print("\tInference...")
                height, width = int(x_buffer_raw.shape[1]), int(x_buffer_raw.shape[2])
                num_h = int((np.ceil(height / FLAGS.patch_size)) + 1)
                num_w = int((np.ceil(width / FLAGS.patch_size)) + 1)
                x_buffer = np.empty(shape=(1, num_h * FLAGS.patch_size, num_w * FLAGS.patch_size, 3))
                y_buffer = np.empty(shape=(num_h * FLAGS.patch_size, num_w * FLAGS.patch_size, 3))
                x_buffer[:,:height, :width, :] = x_buffer_raw
                #y_predict = np.ones(shape=x_buffer.shape)
                for h_index in range(num_h):
                    for w_index in range(num_w):
                        h_start = h_index * FLAGS.patch_size
                        w_start = w_index * FLAGS.patch_size
                        h_end = h_start + FLAGS.patch_size
                        w_end = w_start + FLAGS.patch_size
                        x_buffer_slice = np.float32(x_buffer[:,h_start:h_end, w_start:w_end,:])
                        feed_dict = {x: x_buffer_slice}
                        y = sess.run([y], feed_dict=feed_dict) #(1,256,256,3)
                        y_buffer[h_start:h_end, w_start:w_end,:] = np.squeeze(y) #(256,256,3)
                        #print y_buffer.shape
                y_predict = y_buffer[:height, :width, :]

                # Write to disc
                print_("\tWriting...")
                img_io.write_img(image_sdr_raw, '%s/%03d_in.png' % (FLAGS.out_dir, i+1))
                img_io.write_img(y_predict, '%s/%03d_out.png' % (FLAGS.out_dir, i+1))
                print_("\tWrite image %d done\n" %(i+1) )

            except img_io.IOException as e:
                print_("\n\t\tWarning! ", 'w', True)
                print_("%s\n"%e, 'w')
            except Exception as e:
                print_("\n\t\tError: ", 'e', True)
                print_("%s\n"%e, 'e')

        print_("Done!\n")

