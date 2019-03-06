#!/usr/bin/env python3

from config import get_logging_config, args, train_dir
from config import config as net_config

import time
import os
import sys
import socket
import logging
import logging.config
import subprocess

import keras as K
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')

from vgg import VGG
from resnet import ResNet
from utils import print_variables
from utils_tf import yxyx_to_xywh, data_augmentation
from datasets import get_dataset
from boxer import PriorBoxGrid

slim = tf.contrib.slim
streaming_mean_iou = tf.contrib.metrics.streaming_mean_iou

logging.config.dictConfig(get_logging_config(args.run_name))
log = logging.getLogger()


def objective(location, confidence, refine_ph, classes_ph,
              pos_mask, seg_logits, seg_gt, dataset, config):
    def smooth_l1(x, y):
        abs_diff = tf.abs(x-y)
        return tf.reduce_sum(tf.where(abs_diff < 1,
                                      0.5*abs_diff*abs_diff,
                                      abs_diff - 0.5),
                             1)

    def segmentation_loss(seg_logits, seg_gt, config):
        mask = seg_gt <= dataset.num_classes
        seg_logits = tf.boolean_mask(seg_logits, mask)
        seg_gt = tf.boolean_mask(seg_gt, mask)
        seg_predictions = tf.argmax(seg_logits, axis=1)

        seg_loss_local = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_logits,
                                                                        labels=seg_gt)
        seg_loss = tf.reduce_mean(seg_loss_local)
        tf.summary.scalar('loss/segmentation', seg_loss)

        mean_iou, update_mean_iou = streaming_mean_iou(seg_predictions, seg_gt,
                                                       dataset.num_classes)
        tf.summary.scalar('accuracy/mean_iou', mean_iou)
        return seg_loss, mean_iou, update_mean_iou

    def detection_loss(location, confidence, refine_ph, classes_ph, pos_mask):
        neg_mask = tf.logical_not(pos_mask)
        number_of_positives = tf.reduce_sum(tf.to_int32(pos_mask))
        true_number_of_negatives = tf.minimum(3 * number_of_positives,
                                              tf.shape(pos_mask)[1] - number_of_positives)
        # max is to avoid the case where no positive boxes were sampled
        number_of_negatives = tf.maximum(1, true_number_of_negatives)
        num_pos_float = tf.to_float(tf.maximum(1, number_of_positives))
        normalizer = tf.to_float(tf.add(number_of_positives, number_of_negatives))
        tf.summary.scalar('batch/size', normalizer)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=confidence,
                                                                       labels=classes_ph)
        pos_class_loss = tf.reduce_sum(tf.boolean_mask(cross_entropy, pos_mask))
        tf.summary.scalar('loss/class_pos', pos_class_loss / num_pos_float)

        top_k_worst, top_k_inds = tf.nn.top_k(tf.boolean_mask(cross_entropy, neg_mask),
                                              number_of_negatives)
        # multiplication is to avoid the case where no positive boxes were sampled
        neg_class_loss = tf.reduce_sum(top_k_worst) * \
                         tf.cast(tf.greater(true_number_of_negatives, 0), tf.float32)
        class_loss = (neg_class_loss + pos_class_loss) / num_pos_float
        tf.summary.scalar('loss/class_neg', neg_class_loss / tf.to_float(number_of_negatives))
        tf.summary.scalar('loss/class', class_loss)

        # cond is to avoid the case where no positive boxes were sampled
        bbox_loss = tf.cond(tf.equal(tf.reduce_sum(tf.cast(pos_mask, tf.int32)), 0),
                            lambda: 0.0,
                            lambda: tf.reduce_mean(smooth_l1(tf.boolean_mask(location, pos_mask),
                                                             tf.boolean_mask(refine_ph, pos_mask))))
        tf.summary.scalar('loss/bbox', bbox_loss)

        inferred_class = tf.cast(tf.argmax(confidence, 2), tf.int32)
        positive_matches = tf.equal(tf.boolean_mask(inferred_class, pos_mask),
                                    tf.boolean_mask(classes_ph, pos_mask))
        hard_matches = tf.equal(tf.boolean_mask(inferred_class, neg_mask),
                                tf.boolean_mask(classes_ph, neg_mask))
        hard_matches = tf.gather(hard_matches, top_k_inds)
        train_acc = ((tf.reduce_sum(tf.to_float(positive_matches)) +
                     tf.reduce_sum(tf.to_float(hard_matches))) / normalizer)
        tf.summary.scalar('accuracy/train', train_acc)

        recognized_class = tf.argmax(confidence, 2)
        tp = tf.reduce_sum(tf.to_float(tf.logical_and(recognized_class > 0, pos_mask)))
        fp = tf.reduce_sum(tf.to_float(tf.logical_and(recognized_class > 0, neg_mask)))
        fn = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(recognized_class, 0), pos_mask)))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2*(precision * recall)/(precision + recall)
        tf.summary.scalar('metrics/train/precision', precision)
        tf.summary.scalar('metrics/train/recall', recall)
        tf.summary.scalar('metrics/train/f1', f1)
        return class_loss, bbox_loss, train_acc, number_of_positives

        def instance_loss_graph(target_masks, target_class_ids, pred_masks):
            """Mask binary cross-entropy loss for the masks head.

            target_masks: [batch, num_rois, height, width].
                A float32 tensor of values 0 or 1. Uses zero padding to fill array.
            target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
            pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                        with values from 0 to 1.
            """
            # Reshape for simplicity. Merge first two dimensions into one.
            target_class_ids = K.reshape(target_class_ids, (-1,))
            mask_shape = tf.shape(target_masks)
            target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
            pred_shape = tf.shape(pred_masks)
            pred_masks = K.reshape(pred_masks,
                                   (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
            # Permute predicted masks to [N, num_classes, height, width]
            pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

            # Only positive ROIs contribute to the loss. And only
            # the class specific mask of each ROI.
            positive_ix = tf.where(target_class_ids > 0)[:, 0]
            positive_class_ids = tf.cast(
                tf.gather(target_class_ids, positive_ix), tf.int64)
            indices = tf.stack([positive_ix, positive_class_ids], axis=1)

            # Gather the masks (predicted and true) that contribute to loss
            y_true = tf.gather(target_masks, positive_ix)
            y_pred = tf.gather_nd(pred_masks, indices)

            # Compute binary cross entropy. If no positive ROIs, then return 0.
            # shape: [batch, roi, num_classes]
            loss = K.switch(tf.size(y_true) > 0,
                            K.binary_crossentropy(target=y_true, output=y_pred),
                            tf.constant(0.0))
            loss = K.mean(loss)
            return loss


    the_loss = 0
    train_acc = tf.constant(1)
    mean_iou = tf.constant(1)
    update_mean_iou = tf.constant(1)

    if args.segment:
        seg_loss, mean_iou, update_mean_iou = segmentation_loss(seg_logits, seg_gt, config)
        the_loss += seg_loss

    if args.detect:
        class_loss, bbox_loss, train_acc, number_of_positives =\
            detection_loss(location, confidence, refine_ph, classes_ph, pos_mask)
        det_loss = class_loss + bbox_loss
        the_loss += det_loss

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    wd_loss = tf.add_n(regularization_losses)
    tf.summary.scalar('loss/weight_decay', wd_loss)
    the_loss += wd_loss

    tf.summary.scalar('loss/full', the_loss)
    return the_loss, train_acc, mean_iou, update_mean_iou


def extract_batch(dataset, config):
    with tf.device("/cpu:0"):
        bboxer = PriorBoxGrid(config)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, num_readers=2,
            common_queue_capacity=512, common_queue_min=32)
        if args.instance and args.segment:
            im, bbox, gt, seg, ins, im_h, im_w = \
                data_provider.get(['image', 'object/bbox', 'object/label',
                                   'image/segmentation',
                                   'image/instance',
                                   'image/height',
                                   'image/width'])
            print("ins: ", ins)
            print("bbox: ", bbox)
            print("gt: ", gt)
            ins_shape = tf.stack([im_h, im_w, args.instance_num], axis=0)
            ins = tf.reshape(ins, ins_shape)

        elif args.segment:
            im, bbox, gt, seg = data_provider.get(['image', 'object/bbox', 'object/label',
                                                   'image/segmentation'])
        elif args.instance:
            im, bbox, gt, ins, im_h, im_w = \
                data_provider.get(['image', 'object/bbox', 'object/label',
                                   'image/instance',
                                   'image/height',
                                   'image/width'])
            ins_shape = tf.stack([im_h, im_w, args.instance_num], axis=0)
            ins = tf.reshape(ins, ins_shape)

        else:
            im, bbox, gt = data_provider.get(['image', 'object/bbox', 'object/label'])
            seg = tf.expand_dims(tf.zeros(tf.shape(im)[:2]), 2)
            ins = tf.expand_dims(tf.zeros(tf.shape(im)[:2]), 2)
        im = tf.to_float(im)/255
        bbox = yxyx_to_xywh(tf.clip_by_value(bbox, 0.0, 1.0))
        im, bbox, gt, seg, ins = data_augmentation(im, bbox, gt, seg, ins, config)
        inds, cats, refine, gt_matches = bboxer.encode_gt_tf(bbox, gt)
        bbox_pads = args.instance_num - tf.shape(bbox)[0]
        bbox = tf.pad(bbox, [0, bbox_pads], [0, 0])
        return tf.train.shuffle_batch([im, inds, refine, cats, seg, ins, gt_matches, bbox],
                                      args.batch_size, 2048, 64, num_threads=4)


def train(dataset, net, config):
    image_ph, inds_ph, refine_ph, classes_ph, seg_gt, ins, gt_matches, bbox = extract_batch(dataset, config)
    print("gt_matches:", gt_matches)

    net.create_trunk(image_ph)

    if args.detect:
        net.create_multibox_head(dataset.num_classes)
        confidence = net.outputs['confidence']
        location = net.outputs['location']
        tf.summary.histogram('location', location)
        tf.summary.histogram('confidence', confidence)
    else:
        location, confidence = None, None

    if args.segment:
        net.create_segmentation_head(dataset.num_classes)
        seg_logits = net.outputs['segmentation']
        tf.summary.histogram('segmentation', seg_logits)
    else:
        seg_logits = None

    if args.instance:
        rois = net.top_k_rois  # [batch, num_rois (layer y1 x1 y2 x2)]
        top_k_inds = net.top_k_inds  # [batch, rois_idxs]
        gt_inds = tf.gather_nd(gt_matches, top_k_inds)
        gt_bbox = tf.gather_nd(bbox, gt_inds)
        gt_ins = tf.gather_nd(ins, gt_inds)
        instance_output = net.create_instance_head(dataset.num_classes, rois)

    loss, train_acc, mean_iou, update_mean_iou = objective(location, confidence, refine_ph,
                                                           classes_ph, inds_ph, seg_logits,
                                                           seg_gt, dataset, config)

    ### setting up the learning rate ###
    global_step = slim.get_or_create_global_step()
    learning_rate = args.learning_rate

    learning_rates = [args.warmup_lr, learning_rate]
    steps = [args.warmup_step]

    if len(args.lr_decay) > 0:
        for i, step in enumerate(args.lr_decay):
            steps.append(step)
            learning_rates.append(learning_rate*10**(-i-1))

    learning_rate = tf.train.piecewise_constant(tf.to_int32(global_step),
                                                steps, learning_rates)

    tf.summary.scalar('learning_rate', learning_rate)
    #######

    if args.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif args.optimizer == 'nesterov':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError

    train_vars = tf.trainable_variables()
    print_variables('train', train_vars)

    train_op = slim.learning.create_train_op(
        loss, opt,
        global_step=global_step,
        variables_to_train=train_vars,
        summarize_gradients=True)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000, keep_checkpoint_every_n_hours=1)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.random_trunk_init:
            print("Training from scratch")
        else:
            init_assign_op, init_feed_dict, init_vars = net.get_imagenet_init(opt)
            print_variables('init from ImageNet', init_vars)
            sess.run(init_assign_op, feed_dict=init_feed_dict)

        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if args.ckpt == 0:
                ckpt_to_restore = ckpt.model_checkpoint_path
            else:
                ckpt_to_restore = train_dir+'/model.ckpt-%i' % args.ckpt
            log.info("Restoring model %s..." % ckpt_to_restore)
            saver.restore(sess, ckpt_to_restore)

        starting_step = sess.run(global_step)
        tf.get_default_graph().finalize()
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        log.info("Launching prefetch threads")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        log.info("Starting training...")
        for step in range(starting_step, args.max_iterations+1):
            start_time = time.time()
            try:
                train_loss, acc, iou, _, lr = sess.run([train_op, train_acc, mean_iou,
                                                        update_mean_iou, learning_rate])
            except (tf.errors.OutOfRangeError, tf.errors.CancelledError):
                break
            duration = time.time() - start_time

            num_examples_per_step = args.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('step %d, loss = %.2f, acc = %.2f, iou=%f, lr=%.3f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            log.info(format_str % (step, train_loss, acc, iou, -np.log10(lr),
                                examples_per_sec, sec_per_batch))

            output_inds, output_bbox, output_ins = \
                sess.run([gt_inds, gt_bbox, gt_ins])
            print("output inds:", output_inds)
            print("output bbox:", output_bbox)
            print("output ins:", output_ins)

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 and step > 0:
                summary_writer.flush()
                log.debug("Saving checkpoint...")
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        summary_writer.close()

        coord.request_stop()
        coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
    assert args.detect or args.segment, "Either detect or segment should be True"
    if args.trunk == 'resnet50':
        net = ResNet
        depth = 50
    if args.trunk == 'vgg16':
        net = VGG
        depth = 16

    net = net(config=net_config, depth=depth, training=True, weight_decay=args.weight_decay)

    if args.dataset == 'voc07':
        dataset = get_dataset('voc07_trainval')
    if args.dataset == 'voc12-trainval':
        dataset = get_dataset('voc12-train-segmentation', 'voc12-val')
    if args.dataset == 'voc12-train':
        dataset = get_dataset('voc12-train-segmentation')
    if args.dataset == 'voc12-val':
        dataset = get_dataset('voc12-val-segmentation')
    if args.dataset == 'voc07+12':
        dataset = get_dataset('voc07_trainval', 'voc12_train', 'voc12_val')
    if args.dataset == 'voc07+12-segfull':
        dataset = get_dataset('voc07-trainval-segmentation', 'voc12-train-segmentation', 'voc12-val')
    if args.dataset == 'voc07+12-segmentation':
        dataset = get_dataset('voc07-trainval-segmentation', 'voc12-train-segmentation')
    if args.dataset == 'coco':
        # support by default for coco trainval35k split
        dataset = get_dataset('coco-train2014-*', 'coco-valminusminival2014-*')
    if args.dataset == 'coco-seg':
        # support by default for coco trainval35k split
        dataset = get_dataset('coco-seg-train2014-*')  # , 'coco-seg-val2014-*')  # 'coco-seg-valminusminival2014-*')

    train(dataset, net, net_config)

if __name__ == '__main__':
    exec_string = ' '.join(sys.argv)
    log.debug("Executing a command: %s", exec_string)
    cur_commit = subprocess.check_output("git log -n 1 --pretty=format:\"%H\"".split())
    cur_branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD".split())
    git_diff = subprocess.check_output('git diff --no-color'.split()) #.decode('ascii')
    log.debug("on branch %s with the following diff from HEAD (%s):" % (cur_branch, cur_commit))
    log.debug(git_diff)
    hostname = socket.gethostname()
    if 'gpuhost' in hostname:
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        nvidiasmi = subprocess.check_output('nvidia-smi') #.decode('ascii')
        log.debug("Currently we are on %s and use gpu%s:" % (hostname, gpu_id))
        log.debug(nvidiasmi)
    tf.app.run()
