 # -*- coding = utf-8 -*-
import numpy as np
import tensorflow as tf
import time, os, cv2, random
from scipy.misc import imread,imresize
from os.path import join


SDR_DIR = '../train_data/sdr/'
train_queue = 'new_train.tfrecords'
patch_per_img = 20
patch_size = 256
batch_size = 12

def normalize_imgs_fn(x, type):
    if type == 'sdr':
        x = x*(1./255.)
    else:
        x = x * (1./65535.)
    return x

def data_aug(image, label):
    flag = random.randint(1, 2)
    if flag == 1:
        index = random.randint(-1,1)
        image = cv2.flip(image, index)
        label = cv2.flip(label, index)
    else :
        image, label = image, label
    return image, label

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_to_tfrecords(path, train_queue):
    filename = train_queue
    print('Writting',filename)
    writer = tf.python_io.TFRecordWriter(filename)

    filenames = [name for name in sorted(os.listdir(path))][:12]
    num_files = len(filenames)
    total_num = num_files*patch_per_img
    index = 0
    for i,filename in enumerate(filenames):
        img_path = join(path,filename)
        label_path = img_path.replace('sdr','hdr')
        print 'now is %d of %d :'%(index,total_num)
        # Read JPEG LDR image
        input_image_sdr = cv2.imread(img_path)
        # Read HDR ground truth
        input_image_hdr = cv2.imread(label_path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)

        input_image_sdr, input_image_hdr = data_aug(input_image_sdr, input_image_hdr)
        # convert image type to float32 (conv function needed)
        # input_image_sdr = np.float32(input_image_sdr)
        # input_image_hdr = np.float32(input_image_hdr)
        # # normalize image and label to [0, 1]
        # input_image_sdr = normalize_imgs_fn(input_image_sdr, 'sdr')
        # input_image_hdr = normalize_imgs_fn(input_image_hdr, 'hdr') 
        for step in range(patch_per_img):
            index += 1
            in_row_ind = random.randint(0, input_image_sdr.shape[0] - patch_size)
            in_col_ind = random.randint(0, input_image_sdr.shape[1] - patch_size)
            input_sdr_cropped = input_image_sdr[in_row_ind:in_row_ind + patch_size,
                                                in_col_ind:in_col_ind + patch_size]
            input_hdr_cropped = input_image_hdr[in_row_ind:in_row_ind + patch_size,
                                                in_col_ind:in_col_ind + patch_size]
            cv2.imwrite('crop_i_%d.png' %step, input_sdr_cropped)
            cv2.imwrite('crop_l_%d.png' %step, input_hdr_cropped)

            img_raw = input_sdr_cropped.tostring()
            label_raw = input_hdr_cropped.tostring()
            example=tf.train.Example(features=tf.train.Features(feature={ 
                    'label_raw':  _bytes_feature(label_raw),
                    'image_raw': _bytes_feature(img_raw),
                }))
            writer.write(example.SerializeToString())       
    writer.close()
    print('Writting End')

def decode_from_tfrecords(filename, batch_size = batch_size):
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label_raw' : tf.FixedLenFeature([],tf.string),
                                           'image_raw' : tf.FixedLenFeature([], tf.string)
                                       })

    image = tf.decode_raw(features['image_raw'],tf.uint8)
    label = tf.decode_raw(features['label_raw'],tf.uint16)
    # image,label = data_aug_rot(image,label)
    # image,label = data_aug_flip_ud(image,label)
    # image,label = data_aug_flip_lr(image,label)
    image = tf.reshape(image,[patch_size,patch_size,3])
    label = tf.reshape(label,[patch_size,patch_size,3]) 
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    min_after_dequeue = 1000
    capacity = min_after_dequeue+3*batch_size
    image,label = tf.train.shuffle_batch([image,label],
                             batch_size=batch_size,
                             num_threads=3,
                             capacity=capacity,
                             min_after_dequeue=1000)
    return image,label

def main():
    # convert to tfrecords
    print('convert to train_tfrecords begin')
    start_time = time.time()
    encode_to_tfrecords(SDR_DIR,'new_train')
    duration = time.time() - start_time
    print('convert to train_tfrecords end , cost %d sec' %duration)

    # image,label=decode_from_tfrecords([train_queue],batch_size)
    # image = tf.reshape(image,[batch_size,patch_size,patch_size,3]) 
    # label = tf.reshape(label,[batch_size,patch_size,patch_size,3]) 
    # with tf.Session() as sess: 
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)
    #     coord=tf.train.Coordinator()
    #     threads= tf.train.start_queue_runners(coord=coord)
    #     for i in range(1):
    #         example, l = sess.run([image,label])
    #         cv2.imwrite('image.png',example[0])
    #         l = l.astype(np.uint16)
    #         cv2.imwrite('label.png',l[0])
    #         print(example.shape, l.shape)
    #     coord.request_stop()
    #     coord.join(threads)   

main()



