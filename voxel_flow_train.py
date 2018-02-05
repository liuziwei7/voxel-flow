"""Train a voxel flow model on ucf101 dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataset
from utils.prefetch_queue_shuffle import PrefetchQueue
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
import random
from random import shuffle
from voxel_flow_model import Voxel_flow_model
from utils.image_utils import imwrite
from functools import partial
import pdb

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_string('train_dir', './voxel_flow_checkpoints/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_image_dir', './voxel_flow_train_image/',
			   """Directory where to output images.""")
tf.app.flags.DEFINE_string('test_image_dir', './voxel_flow_test_image/',
			   """Directory where to output images.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', './voxel_flow_checkpoints/',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer(
        'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0003,
                          """Initial learning rate.""")


def train(dataset_frame1, dataset_frame2, dataset_frame3):
  """Trains a model."""
  with tf.Graph().as_default():
    # Create input and target placeholder.
    input_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, 6))
    target_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, 3))

    # input_resized = tf.image.resize_area(input_placeholder, [128, 128])
    # target_resized = tf.image.resize_area(target_placeholder,[128, 128])

    # Prepare model.
    model = Voxel_flow_model()
    prediction = model.inference(input_placeholder) d
    # reproduction_loss, prior_loss = model.loss(prediction, target_placeholder)
    reproduction_loss = model.loss(prediction, target_placeholder)
    # total_loss = reproduction_loss + prior_loss
    total_loss = reproduction_loss
    
    # Perform learning rate scheduling.
    learning_rate = FLAGS.initial_learning_rate

    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(learning_rate)
    grads = opt.compute_gradients(total_loss)
    update_op = opt.apply_gradients(grads)

    # Create summaries
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summaries.append(tf.scalar_summary('total_loss', total_loss))
    summaries.append(tf.scalar_summary('reproduction_loss', reproduction_loss))
    # summaries.append(tf.scalar_summary('prior_loss', prior_loss))
    summaries.append(tf.image_summary('Input Image', input_placeholder, 3))
    summaries.append(tf.image_summary('Output Image', prediction, 3))
    summaries.append(tf.image_summary('Target Image', target_placeholder, 3))

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # Summary Writter
    summary_writer = tf.train.SummaryWriter(
      FLAGS.train_dir,
      graph=sess.graph)

    # Training loop using feed dict method.
    data_list_frame1 = dataset_frame1.read_data_list_file()
    random.seed(1)
    shuffle(data_list_frame1)

    data_list_frame2 = dataset_frame2.read_data_list_file()
    random.seed(1)
    shuffle(data_list_frame2)

    data_list_frame3 = dataset_frame3.read_data_list_file()
    random.seed(1)
    shuffle(data_list_frame3)

    data_size = len(data_list_frame1)
    epoch_num = int(data_size / FLAGS.batch_size)

    # num_workers = 1
      
    # load_fn_frame1 = partial(dataset_frame1.process_func)
    # p_queue_frame1 = PrefetchQueue(load_fn_frame1, data_list_frame1, FLAGS.batch_size, shuffle=False, num_workers=num_workers)

    # load_fn_frame2 = partial(dataset_frame2.process_func)
    # p_queue_frame2 = PrefetchQueue(load_fn_frame2, data_list_frame2, FLAGS.batch_size, shuffle=False, num_workers=num_workers)

    # load_fn_frame3 = partial(dataset_frame3.process_func)
    # p_queue_frame3 = PrefetchQueue(load_fn_frame3, data_list_frame3, FLAGS.batch_size, shuffle=False, num_workers=num_workers)

    for step in xrange(0, FLAGS.max_steps):
      batch_idx = step % epoch_num
      
      batch_data_list_frame1 = data_list_frame1[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      batch_data_list_frame2 = data_list_frame2[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      batch_data_list_frame3 = data_list_frame3[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      
      # Load batch data.
      batch_data_frame1 = np.array([dataset_frame1.process_func(line) for line in batch_data_list_frame1])
      batch_data_frame2 = np.array([dataset_frame2.process_func(line) for line in batch_data_list_frame2])
      batch_data_frame3 = np.array([dataset_frame3.process_func(line) for line in batch_data_list_frame3])

      # batch_data_frame1 = p_queue_frame1.get_batch()
      # batch_data_frame2 = p_queue_frame2.get_batch()
      # batch_data_frame3 = p_queue_frame3.get_batch()

      feed_dict = {input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3), target_placeholder: batch_data_frame2}
     
      # Run single step update.
      _, loss_value = sess.run([update_op, total_loss], feed_dict = feed_dict)
      
      if batch_idx == 0:
        # Shuffle data at each epoch.
        random.seed(1)
        shuffle(data_list_frame1)
        random.seed(1)
        shuffle(data_list_frame2)
        random.seed(1)
        shuffle(data_list_frame3)
        print('Epoch Number: %d' % int(step / epoch_num))
      
      # Output Summary 
      if step % 10 == 0:
        # summary_str = sess.run(summary_op, feed_dict = feed_dict)
        # summary_writer.add_summary(summary_str, step)
	      print("Loss at step %d: %f" % (step, loss_value))

      if step % 500 == 0:
        # Run a batch of images	
        prediction_np, target_np = sess.run([prediction, target_placeholder], feed_dict = feed_dict) 
        for i in range(0,prediction_np.shape[0]):
          file_name = FLAGS.train_image_dir+str(i)+'_out.png'
          file_name_label = FLAGS.train_image_dir+str(i)+'_gt.png'
          imwrite(file_name, prediction_np[i,:,:,:])
          imwrite(file_name_label, target_np[i,:,:,:])

      # Save checkpoint 
      if step % 5000 == 0 or (step +1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def validate(dataset_frame1, dataset_frame2, dataset_frame3):
  """Performs validation on model.
  Args:  
  """
  pass

def test(dataset_frame1, dataset_frame2, dataset_frame3):
  """Perform test on a trained model."""
  with tf.Graph().as_default():
		# Create input and target placeholder.
    input_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, 6))
    target_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, 3))
    
    # input_resized = tf.image.resize_area(input_placeholder, [128, 128])
    # target_resized = tf.image.resize_area(target_placeholder,[128, 128])

    # Prepare model.
    model = Voxel_flow_model(is_train=True)
    prediction = model.inference(input_placeholder)
    # reproduction_loss, prior_loss = model.loss(prediction, target_placeholder)
    reproduction_loss = model.loss(prediction, target_placeholder)
    # total_loss = reproduction_loss + prior_loss
    total_loss = reproduction_loss

    # Create a saver and load.
    saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()

    # Restore checkpoint from file.
    if FLAGS.pretrained_model_checkpoint_path:
      assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      ckpt = tf.train.get_checkpoint_state(
               FLAGS.pretrained_model_checkpoint_path)
      restorer = tf.train.Saver()
      restorer.restore(sess, ckpt.model_checkpoint_path)
      print('%s: Pre-trained model restored from %s' %
        (datetime.now(), ckpt.model_checkpoint_path))
    
    # Process on test dataset.
    data_list_frame1 = dataset_frame1.read_data_list_file()
    data_size = len(data_list_frame1)
    epoch_num = int(data_size / FLAGS.batch_size)

    data_list_frame2 = dataset_frame2.read_data_list_file()

    data_list_frame3 = dataset_frame3.read_data_list_file()

    i = 0 
    PSNR = 0

    for id_img in range(0, data_size):  
      # Load single data.
      line_image_frame1 = dataset_frame1.process_func(data_list_frame1[id_img])
      line_image_frame2 = dataset_frame2.process_func(data_list_frame2[id_img])
      line_image_frame3 = dataset_frame3.process_func(data_list_frame3[id_img])
      
      batch_data_frame1 = [dataset_frame1.process_func(ll) for ll in data_list_frame1[0:63]]
      batch_data_frame2 = [dataset_frame2.process_func(ll) for ll in data_list_frame2[0:63]]
      batch_data_frame3 = [dataset_frame3.process_func(ll) for ll in data_list_frame3[0:63]]
      
      batch_data_frame1.append(line_image_frame1)
      batch_data_frame2.append(line_image_frame2)
      batch_data_frame3.append(line_image_frame3)
      
      batch_data_frame1 = np.array(batch_data_frame1)
      batch_data_frame2 = np.array(batch_data_frame2)
      batch_data_frame3 = np.array(batch_data_frame3)
      
      feed_dict = {input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame3), 3),
                    target_placeholder: batch_data_frame2}
      # Run single step update.
      prediction_np, target_np, loss_value = sess.run([prediction,
                                                      target_placeholder,
                                                      total_loss],
                                                      feed_dict = feed_dict)
      print("Loss for image %d: %f" % (i,loss_value))
      file_name = FLAGS.test_image_dir+str(i)+'_out.png'
      file_name_label = FLAGS.test_image_dir+str(i)+'_gt.png'
      imwrite(file_name, prediction_np[-1,:,:,:])
      imwrite(file_name_label, target_np[-1,:,:,:])
      i += 1
      PSNR += 10*np.log10(255.0*255.0/np.sum(np.square(prediction_np-target_np)))
    print("Overall PSNR: %f db" % (PSNR/len(data_list)))
      
if __name__ == '__main__':
  
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  if FLAGS.subset == 'train':
    
    data_list_path_frame1 = "data_list/ucf101_train_files_frame1.txt"
    data_list_path_frame2 = "data_list/ucf101_train_files_frame2.txt"
    data_list_path_frame3 = "data_list/ucf101_train_files_frame3.txt"
    
    ucf101_dataset_frame1 = dataset.Dataset(data_list_path_frame1) 
    ucf101_dataset_frame2 = dataset.Dataset(data_list_path_frame2) 
    ucf101_dataset_frame3 = dataset.Dataset(data_list_path_frame3) 
    
    train(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3)
  
  elif FLAGS.subset == 'test':
    
    data_list_path_frame1 = "data_list/ucf101_test_files_frame1.txt"
    data_list_path_frame2 = "data_list/ucf101_test_files_frame2.txt"
    data_list_path_frame3 = "data_list/ucf101_test_files_frame3.txt"
    
    ucf101_dataset_frame1 = dataset.Dataset(data_list_path_frame1) 
    ucf101_dataset_frame2 = dataset.Dataset(data_list_path_frame2) 
    ucf101_dataset_frame3 = dataset.Dataset(data_list_path_frame3) 
    
    test(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3)
