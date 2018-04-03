"""
 " Description: TensorFlow ExpandNet for HDR image reconstruction.
 " Author: LiJinghui
 " Date: March 2018
""" 
import  tensorflow as tf  
from  tf_create import  decode_from_tfrecords 
from datetime import datetime
import  cv2, os
import model, img_io
from img_io import map_range 
 
log_dir = '../train_output/logs' 
im_dir = '../train_output/im'
train_queue = 'new_train.tfrecords'
start_learning_rate = 7e-5
lr_update_step = 10000
batch_size=12
max_iter=100000000
patch_size = 256
lambda_ = 5

def train():   
    train_queue = [train_queue]
    input_data,input_target=decode_from_tfrecords(train_queue, batch_size)  
    input_data = np.float32(tf.reshape(input_data,[batch_size,patch_size,patch_size,3]))
    input_target = np.float32(tf.reshape(input_target,[batch_size,patch_size,patch_size,3]))
    input_data = map_range(input_data)
    input_target = map_range(input_target)

    logits = model.model(input_data) # (batch_size, 256, 256, 3)
    logits = tf.clip_by_value(logits, 0, 1)

    # === loss with l2_regularize ========================================================
    d_l1 = tf.losses.absolute_difference(logits, input_target)
    d_cosin = tf.losses.cosine_distance(tf.nn.l2_normalize(logits, 3), tf.nn.l2_normalize(input_target, 3), dim=3)
    loss = d_l1 + lambda_ * d_cosin
    tf.add_to_collection('losses', loss)

    regularizer = tf.contrib.layers.l2_regularizer(1e-3) #return a l2 regularize function
    reg_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=None) #apply l2_regularizer,return a scale tensor
    tf.add_to_collection('losses', reg_loss)
    loss = tf.add_n(tf.get_collection('losses'))

    tf.scalar_summary("cost_function", loss)
    
    # Adam optimizer and learning rate exponential_decay
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               int(lr_update_step), 0.8, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                          epsilon=1e-8, use_locking=False).minimize(loss, global_step = global_step)

    tf.scalar_summary("learning_rate", learning_rate)
    merged_summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver()

    with tf.Session() as session:  
        summary_writer = tf.summary.FileWriter(log_dir, session.graph)
        init=tf.global_variables_initializer()
        session.run(init)  

        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)

        model_path = os.path.join("../model_pre",'model.ckpt')
        if os.path.exists(model_path) is True:  
            model_file=tf.train.latest_checkpoint(model_path)
            saver.restore(session,model_file)
        tf.logging.set_verbosity(tf.logging.INFO)   

        loss_cnt = 0.0
        for step in range(max_iters):
            _, loss_value, summary_str, output = session.run([train_op, loss, merged_summary_op, logits])
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            loss_cnt+=loss_value

            summary_writer.add_summary(summary_str, step) 

            if step % 100 == 0:
                print('-------------------------------------------')
                print('Currently at step %d of %d.' % (step, max_iters))
                print('%s: loss on training batch is %.5f' % (datetime.now(), total_loss/100.0))
                print('\n')
                loss_cnt = 0.0

            if step % 1000 == 0:
                x = np.squeeze(input_data[0])
                y_gt = np.squeeze(input_target[0])
                y_out = np.squeeze(output[0])
                img_io.write_LDR(x, "%s/%06d_in.png" % (im_dir, step))
                img_io.write_HDR(y_gt, "%s/%06d_gt.png" % (im_dir, step))
                img_io.write_HDR(y_out, "%s/%06d_out.png" % (im_dir, step))

            if step % 5000 == 0 or (step + 1) == max_iters:
                saver.save(session, model_path, global_step=step)
        
        coord.request_stop()
        coord.join(threads)  

train()