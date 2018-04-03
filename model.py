"""
 " Description: TensorFlow ExpandNet for HDR image reconstruction.
 " Author: LiJinghui
 " Date: March 2018
"""
import tensorflow as tf

# The HDR reconstruction autoencoder fully convolutional neural network
def model(net_in):
    # net_in (batch_size, 256, 256, 3)
    loc = local_net(net_in)#(batch_size, 256, 256, 128)
    mid = mid_net(net_in)#(batch_size, 256, 256, 64)
    glob = glob_net(net_in)#(1, 256, 256, 64)
    exp_glob = expand_g(glob,shape = (1,256,256,1))
    fuse = tf.concat((loc, mid, exp_glob),3)#(batch_size, 256, 256, 256)
    network = end_net(fuse)#(batch_size, 256, 256, 3)
    return network

def layer(name, inputs, shape, strides, padding, apply_selu, atrous_rate=None):
        with tf.variable_scope(name):
            tf.cast(inputs, tf.float32)
            # filters = tf.contrib.layers.xavier_initializer()
            filters = get_weight(shape)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filters)
            if not atrous_rate:
                conv_out = tf.nn.conv2d(inputs, filters, strides, padding)
            else:
                conv_out = tf.nn.atrous_conv2d(inputs, filters, atrous_rate, padding)
            if apply_selu:
                conv_out = tf.nn.selu(conv_out)
        return conv_out

def local_net(net_in):
    network = layer('l_conv1', net_in, shape = (3, 3, 3, 64), strides=[1, 1, 1, 1], padding='SAME', apply_selu=True)
    network = layer('l_conv2', network, shape = (3, 3, 64, 128), strides=[1, 1, 1, 1], padding='SAME', apply_selu=True)
    return network

def mid_net(net_in):
    network = layer('m_conv1', net_in, shape = (3, 3, 3, 64), strides=[1, 1, 1, 1], padding='SAME', apply_selu=True, atrous_rate=2)
    network = layer('m_conv2', network, shape = (3, 3, 64, 64), strides=[1, 1, 1, 1], padding='SAME', apply_selu=True, atrous_rate=2)
    network = layer('m_conv3', network, shape = (3, 3, 64, 64), strides=[1, 1, 1, 1], padding='SAME', apply_selu=True, atrous_rate=2)
    network = layer('m_conv3', network, shape = (3, 3, 64, 64), strides=[1, 1, 1, 1], padding='SAME', apply_selu=False, atrous_rate=2)
    return network

def glob_net(net_in):
    network = layer('g_conv1', net_in, shape = (3, 3, 3, 64), strides=[1, 2, 2, 1], padding='SAME', apply_selu=True)
    network = layer('g_conv2', network, shape = (3, 3, 64, 64), strides=[1, 2, 2, 1], padding='SAME', apply_selu=True)
    network = layer('g_conv3', network, shape = (3, 3, 64, 64), strides=[1, 2, 2, 1], padding='SAME', apply_selu=True)
    network = layer('g_conv4', network, shape = (3, 3, 64, 64), strides=[1, 2, 2, 1], padding='SAME', apply_selu=True)
    network = layer('g_conv5', network, shape = (3, 3, 64, 64), strides=[1, 2, 2, 1], padding='SAME', apply_selu=True)
    network = layer('g_conv6', network, shape = (3, 3, 64, 64), strides=[1, 2, 2, 1], padding='SAME', apply_selu=True)
    network = layer('g_conv7', network, shape = (4, 4, 64, 64), strides=[1, 1, 1, 1], padding='VALID', apply_selu=False)
    return network

def end_net(net_in):
    network = layer('e_conv1', net_in, shape = (1, 1, 256, 64), strides=[1, 1, 1, 1], padding='SAME', apply_selu=True)
    network = layer('e_conv2', network, shape = (3, 3, 64, 3), strides=[1, 1, 1, 1], padding='SAME', apply_selu=False)
    network = tf.nn.sigmoid(network)
    return network

def expand_g(net_in,shape):
    output = tf.tile(net_in, shape)
    return output

def get_weight(shape, lambda_=1e-7, stddev=0.09):
    weights_init = tf.Variable(tf.truncated_normal(shape,stddev))
    # regularizer = tf.contrib.layers.l2_regularizer(lambda_)(weights_init)
    # tf.add_to_collection('losses', regularizer)
    return weights_init
