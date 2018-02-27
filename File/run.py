#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 17:16:14 2017

@author: wzg
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
from six.moves import xrange
import scipy.io as sio
import scipy.misc as misc


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "4", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("visualize_dir", "Data/visualize/", "path to visualzie directory")
tf.flags.DEFINE_float("learning_rate", "1e-6", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")

root_path = './Data/'
train_tfrecord_filename = root_path + 'tfrecords/train.tfrecords'
test_tfrecord_filename = root_path + 'tfrecords/test.tfrecords'
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
ckpt_path = './Model/MTV4_model.ckpt'

MAX_ITERATION = 8000
NUM_OF_CLASSESS = 6
WEATHER_CLASSES = 2
IMAGE_SIZE = 300
EPOCHS = 5
RESTORE = True

meanfile = sio.loadmat(root_path + 'mats/mean300.mat')
meanvalue = meanfile['mean']  #mean value of images in training set



def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name = name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name = name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name = name)
            tf.add_to_collection('activations', current)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    weights = np.squeeze(model_data['layers'])

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6_1 = utils.weight_variable([7, 7, 512, 4096], name = "W6_1")
        b6_1 = utils.bias_variable([4096], name = "b6_1")
        conv6_1 = utils.conv2d_basic(pool5, W6_1, b6_1)
        relu6_1 = tf.nn.relu(conv6_1, name = "relu6_1")
        if FLAGS.debug:
            utils.add_activation_summary(relu6_1)
        relu_dropout6_1 = tf.nn.dropout(relu6_1, keep_prob = keep_prob)
           
  

        W7_1 = utils.weight_variable([1, 1, 4096, 4096], name = "W7_1")
        b7_1 = utils.bias_variable([4096], name = "b7_1")
        conv7_1 = utils.conv2d_basic(relu_dropout6_1, W7_1, b7_1)
        relu7_1 = tf.nn.relu(conv7_1, name = "relu7_1")
        if FLAGS.debug:
            utils.add_activation_summary(relu7_1)
        relu_dropout7_1 = tf.nn.dropout(relu7_1, keep_prob = keep_prob)
                

        W8_1 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name = "W8_1")
        b8_1 = utils.bias_variable([NUM_OF_CLASSESS], name = "b8_1")
        conv8_1 = utils.conv2d_basic(relu_dropout7_1, W8_1, b8_1)
        if FLAGS.debug:
            tf.summary.histogram("conv8_1/activation", conv8_1)
            tf.summary.scalar("conv8_1/sparsity", tf.nn.zero_fraction(conv8_1))
        

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name = "W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name = "b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8_1, W_t1, b_t1, output_shape = tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name = "fuse_1")

        if FLAGS.debug:
            tf.summary.histogram("conv_t1/activation", conv_t1)
            tf.summary.scalar("conv_t1/sparsity", tf.nn.zero_fraction(conv_t1))
            tf.summary.histogram("fuse_1/activation", fuse_1)
            tf.summary.scalar("fuse_1/sparsity", tf.nn.zero_fraction(fuse_1))

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name = "W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name = "b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape = tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name = "fuse_2")

        if FLAGS.debug:
            tf.summary.histogram("conv_t2/activation", conv_t2)
            tf.summary.scalar("conv_t2/sparsity", tf.nn.zero_fraction(conv_t2))
            tf.summary.histogram("fuse_2/activation", fuse_2)
            tf.summary.scalar("fuse_2/sparsity", tf.nn.zero_fraction(fuse_2))

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name = "W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name = "b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape = deconv_shape3, stride = 8)

        annotation_pred = tf.argmax(conv_t3, axis = 3, name = "prediction")

        if FLAGS.debug:    
            tf.summary.histogram("conv_t3/activation", conv_t3)
            tf.summary.scalar("conv_t3/sparsity", tf.nn.zero_fraction(conv_t3))
            tf.summary.histogram("annotation_pred/activation", annotation_pred)
            tf.summary.scalar("annotation_pred/sparsity", tf.nn.zero_fraction(annotation_pred))
            
            
            
############################### Another branch of multi-task architecture ################################
        W6_2 = utils.weight_variable([7, 7, 512, 1024], name = "W6_2")
        b6_2 = utils.bias_variable([1024], name = "b6_2")
        conv6_2 = utils.conv2d_basic(pool5, W6_2, b6_2)
        relu6_2 = tf.nn.relu(conv6_2, name = "relu6_2")
        if FLAGS.debug:
            utils.add_activation_summary(relu6_2)
        relu_dropout6_2 = tf.nn.dropout(relu6_2, keep_prob = keep_prob)
        
        W7_2 = utils.weight_variable([1, 1, 1024, 3840], name = "W7_2")
        b7_2 = utils.bias_variable([3840], name = "b7_2")
        conv7_2 = utils.conv2d_basic(relu_dropout6_2, W7_2, b7_2)
        relu7_2 = tf.nn.relu(conv7_2, name = "relu7_2")
        if FLAGS.debug:
            utils.add_activation_summary(relu7_2)
        relu_dropout7_2 = tf.nn.dropout(relu7_2, keep_prob = keep_prob)
        
        kernel_height = conv7_2.get_shape()[1]
        kernel_width = conv7_2.get_shape()[2]
        conv7_2_gapool = tf.nn.avg_pool(relu_dropout7_2, ksize = [1, kernel_height, kernel_width, 1],
                                             strides = [1, kernel_height, kernel_width, 1], padding = "SAME")
        
        kernel_height2 = fuse_2.get_shape()[1]
        kernel_width2 = fuse_2.get_shape()[2]
        fuse_2_gapool = tf.nn.avg_pool(fuse_2, ksize=[1, kernel_height2, kernel_width2, 1],
                                       strides = [1, kernel_height2, kernel_width2, 1], padding = "SAME")
        
        concat_res = tf.concat([conv7_2_gapool, fuse_2_gapool], axis = 3)
        concat_res = tf.squeeze(concat_res)
        
        
        W8_2 = utils.weight_variable([4096, WEATHER_CLASSES], name = "W8_2")
        b8_2 = utils.bias_variable([WEATHER_CLASSES], name = "b8_2")
        logits = tf.nn.bias_add(tf.matmul(concat_res, W8_2), b8_2)
        
        
    return tf.expand_dims(annotation_pred, dim = 3), conv_t3, logits


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list = var_list)
    if FLAGS.debug:
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)



def main(argv = None):
    keep_probability = tf.placeholder(tf.float32, name = "keep_probabilty")
    image = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE, IMAGE_SIZE, 3], name = "input_image")
    annotation = tf.placeholder(tf.int32, shape = [None, IMAGE_SIZE, IMAGE_SIZE, 1], name = "annotation")
    label = tf.placeholder(tf.float32, [None, WEATHER_CLASSES], name = 'label')

    pred_annotation, logits, pred_label = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs = 2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs = 2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs = 2)

    seg_loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims = [3]),
                                                                          name = "seg_loss")))

    classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_label, labels = label,
                                                                                                name = 'classification_loss'))

    regular_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    total_loss = seg_loss + classification_loss + regular_loss
    
    tf.summary.scalar("seg_loss", seg_loss)
    tf.summary.scalar("classification_loss", classification_loss)
    tf.summary.scalar("total_loss", total_loss)
    
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(pred_label, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
    tf.summary.scalar('accuracy', accuracy)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(total_loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
 

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, tf.get_default_graph())
    
    
    print("Setting up dataset reader")
    train_filename_queue = tf.train.string_input_producer([train_tfrecord_filename],
                                                    num_epochs=EPOCHS)
    mytrain_images, mytrain_annotations, mytrain_names, mytrain_labels = utils.read_and_decode(train_filename_queue)
    
    
    test_filename_queue = tf.train.string_input_producer([test_tfrecord_filename],
                                                    num_epochs=EPOCHS)
    mytest_images, mytest_annotations, mytest_names, mytest_labels = utils.read_and_decode(test_filename_queue, shuffle_batch=False)

    

    
    with tf.Session(graph=tf.get_default_graph()) as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
        sess.run(init_op)
        
        if RESTORE:
            saver.restore(sess, ckpt_path)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
    
        if FLAGS.mode == "train":
            for itr in xrange(MAX_ITERATION):
                #print('Iteration: %d' %itr)
                train_images, train_annotations, train_labels = sess.run([mytrain_images, mytrain_annotations, mytrain_labels])
                train_annotations = np.array(np.expand_dims(train_annotations, axis = 3))
                one_hot_labels = utils.get_one_hot_label(FLAGS.batch_size, WEATHER_CLASSES, train_labels)
                feed_dict = {image: train_images, annotation: train_annotations, label: one_hot_labels, keep_probability: 0.50}
    
                sess.run(train_op, feed_dict = feed_dict)
                
                if (itr + 1) % 2000 == 0:
                     train_acc, sum_seg_loss, sum_classification_loss, sum_total_loss = 0, 0, 0, 0
                    
                     ####### Train: 1 epoch == 2000 iteration ##################
                     for j in xrange(2000):
                         train_images, train_annotations, train_labels = sess.run([mytrain_images, mytrain_annotations, mytrain_labels])
                         train_annotations = np.array(np.expand_dims(train_annotations, axis = 3))
                         one_hot_labels = utils.get_one_hot_label(FLAGS.batch_size, WEATHER_CLASSES, train_labels)
                         feed_dict = {image: train_images, annotation: train_annotations, label: one_hot_labels, keep_probability: 1.0}
                        
                         acc, train_seg_loss, train_classification_loss, train_total_loss, \
                         summary_str = sess.run([accuracy, seg_loss, classification_loss, total_loss, summary_op], feed_dict = feed_dict)
                         summary_writer.add_summary(summary_str, itr + 1 - 2000 + j)
                        
                         train_acc += acc
                         sum_seg_loss += train_seg_loss 
                         sum_classification_loss += train_classification_loss
                         sum_total_loss += train_total_loss
                    
                     train_acc /= 2000
                     sum_seg_loss /= 2000
                     sum_classification_loss /= 2000
                     sum_total_loss /= 2000
                     print("Step: %d, Train: accuracy: %g, seg_loss: %g, classification_loss: %g, total_loss: %g" % (itr + 1, train_acc, sum_seg_loss, sum_classification_loss, sum_total_loss))
                    
                     test_acc, sum_seg_loss, sum_classification_loss, sum_total_loss = 0, 0, 0, 0
                    ######### Test: 1 epoch == 500 iteration ##################
                    
                     saver.save(sess, FLAGS.model_dir + "your_model_name.ckpt", itr)
        
        elif FLAGS.mode == "test":
            test_acc, sum_seg_loss, sum_classification_loss, sum_total_loss = 0, 0, 0, 0
            for itr in xrange(500):
                test_images, test_annotations, test_labels = sess.run([mytest_images, mytest_annotations, mytest_labels])
                test_annotations = np.array(np.expand_dims(test_annotations, axis = 3))
                one_hot_labels = utils.get_one_hot_label(FLAGS.batch_size, WEATHER_CLASSES, test_labels)
                feed_dict = {image: test_images, annotation: test_annotations, label: one_hot_labels, keep_probability: 1.0}
                
                acc, test_seg_loss, test_classification_loss, test_total_loss = \
                sess.run([accuracy, seg_loss, classification_loss, total_loss], feed_dict = feed_dict)
                
                test_acc += acc
                sum_seg_loss += test_seg_loss 
                sum_classification_loss += test_classification_loss
                sum_total_loss += test_total_loss
                    
            test_acc /= 500
            sum_seg_loss /= 500
            sum_classification_loss /= 500
            sum_total_loss /= 500
            print("Test: accuracy: %g, seg_loss: %g, classification_loss: %g, total_loss: %g" \
                  % (test_acc, sum_seg_loss, sum_classification_loss, sum_total_loss))
            
        elif FLAGS.mode == "visualize":
            for k in xrange(10):
                test_images, test_annotations = sess.run([mytest_images, mytest_annotations])
                test_annotations = np.array(np.expand_dims(test_annotations, axis = 3))
                pred = sess.run(pred_annotation, feed_dict = {image: test_images, annotation: test_annotations, keep_probability: 1.0})
                test_annotations = np.squeeze(test_annotations, axis = 3)
                pred = np.squeeze(pred, axis = 3)
        
                for itr in range(FLAGS.batch_size):
                    misc.imsave(FLAGS.visualize_dir + "img_" + str(k*FLAGS.batch_size+itr) + '.jpg', (test_images[itr] + meanvalue).astype(np.uint8))
                    sio.savemat(FLAGS.visualize_dir + 'gt_' + str(k*FLAGS.batch_size+itr) + '.mat', {'mask':test_annotations[itr].astype(np.uint8)})
                    sio.savemat(FLAGS.visualize_dir + 'pred_' + str(k*FLAGS.batch_size+itr) + '.mat', {'mask':pred[itr].astype(np.uint8)})
                    print("Saved image: %d" % (k*FLAGS.batch_size+itr))
                
        sess.close()
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()