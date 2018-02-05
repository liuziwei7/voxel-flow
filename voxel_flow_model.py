"""Implements a voxel flow model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.loss_utils import l1_loss, l2_loss, vae_loss 
from utils.geo_layer_utils import vae_gaussian_layer
from utils.geo_layer_utils import bilinear_interp
from utils.geo_layer_utils import meshgrid

class Voxel_flow_model(object):
  def __init__(self, is_train=True):
    self.is_train = is_train

  def inference(self, input_images):
    """Inference on a set of input_images.
    Args:
    """
    return self._build_model(input_images) 

  def loss(self, predictions, targets):
    """Compute the necessary loss for training.
    Args:
    Returns:
    """
    self.reproduction_loss = l1_loss(predictions, targets) #l2_loss(predictions, targets)
    # self.prior_loss = vae_loss(self.z_mean, self.z_logvar, prior_weight = 0.0001)

    # return [self.reproduction_loss, self.prior_loss]
    return self.reproduction_loss

  def _build_model(self, input_images):
    """Build a VAE model.
    Args:
    """

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
      
      # Define network      
      batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'is_training': self.is_train,
      }
      with slim.arg_scope([slim.batch_norm], is_training = self.is_train, updates_collections=None):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
          net = slim.conv2d(input_images, 64, [5, 5], stride=1, scope='conv1')
          net = slim.max_pool2d(net, [2, 2], scope='pool1')
          net = slim.conv2d(net, 128, [5, 5], stride=1, scope='conv2')
          net = slim.max_pool2d(net, [2, 2], scope='pool2')
          net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3')
          net = slim.max_pool2d(net, [2, 2], scope='pool3')
          net = tf.image.resize_bilinear(net, [64,64])
          net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')
          net = tf.image.resize_bilinear(net, [128,128])
          net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv5')
          net = tf.image.resize_bilinear(net, [256,256])
          net = slim.conv2d(net, 64, [5, 5], stride=1, scope='conv6')
    net = slim.conv2d(net, 3, [5, 5], stride=1, activation_fn=tf.tanh,
    normalizer_fn=None, scope='conv7')
    
    flow = net[:, :, :, 0:2]
    mask = tf.expand_dims(net[:, :, :, 2], 3)

    grid_x, grid_y = meshgrid(256, 256)
    grid_x = tf.tile(grid_x, [32, 1, 1]) # batch_size = 32
    grid_y = tf.tile(grid_y, [32, 1, 1]) # batch_size = 32

    flow = 0.5 * flow

    coor_x_1 = grid_x + flow[:, :, :, 0]
    coor_y_1 = grid_y + flow[:, :, :, 1]

    coor_x_2 = grid_x - flow[:, :, :, 0]
    coor_y_2 = grid_y - flow[:, :, :, 1]    
    
    output_1 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1, coor_y_1, 'interpolate')
    output_2 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2, coor_y_2, 'interpolate')

    mask = 0.5 * (1.0 + mask)
    mask = tf.tile(mask, [1, 1, 1, 3])
    net = tf.mul(mask, output_1) + tf.mul(1.0 - mask, output_2)

    return net
