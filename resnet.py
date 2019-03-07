import logging
import os

import tensorflow as tf

import resnet_v1
import resnet_utils

from resnet_v1 import bottleneck, bottleneck_skip
from utils import print_variables
import config
from config import args, MEAN_COLOR
from paths import INIT_WEIGHTS_DIR
import numpy as np
import keras.layers as KL
import keras.engine as KE

log = logging.getLogger()
slim = tf.contrib.slim

CKPT_50 = os.path.join(INIT_WEIGHTS_DIR, 'resnet50_full.ckpt')
DEFAULT_SCOPE_50 = 'resnet_v1_50'
DEFAULT_SSD_SCOPE = 'ssd'


class ResNet(object):
    def __init__(self, config, training, weight_decay=0.0005, depth=50,
                 reuse=False):
        self.config = config
        self.weight_decay = weight_decay
        self.layers = []
        self.reuse = reuse
        self.training = training
        self.layers = self.config['layers']
        self.roi_bounds = ROI_bounds(config)
        (config)
        if depth == 50:
            self.num_block3 = 5
            self.scope = DEFAULT_SCOPE_50
            self.ckpt = CKPT_50
        else:
            raise ValueError

    def create_trunk(self, images):
        red, green, blue = tf.split(images*255, 3, axis=3)
        images = tf.concat([blue, green, red], 3) - MEAN_COLOR

        with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=self.training,
                                                       weight_decay=self.weight_decay,
                                                       batch_norm_decay=args.bn_decay)):
            blocks = [
                resnet_utils.Block(
                    'block1', bottleneck, [(256, 64, 1)] * 3),
                resnet_utils.Block(
                    'block2', bottleneck, [(512, 128, 2)] + [(512, 128, 1)] * 3),
                resnet_utils.Block(
                    'block3', bottleneck, [(1024, 256, 2)] + [(1024, 256, 1)] * self.num_block3),
                resnet_utils.Block(
                    'block4', bottleneck, [(2048, 512, 2)] + [(2048, 512, 1)] * 2)
            ]

            net, endpoints = resnet_v1.resnet_v1(images, blocks,
                                                 global_pool=False,
                                                 reuse=self.reuse,
                                                 scope=self.scope)
            self.outputs = endpoints
        self.add_extra_layers(net)

    def vgg_arg_scope(self):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME') as arg_sc:
            return arg_sc

    def add_extra_layers(self, net):
        with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=self.training,
                                                       weight_decay=self.weight_decay,
                                                       batch_norm_decay=args.bn_decay)):
            block_depth = 2
            num_fm = 2048
            blocks = [
                resnet_utils.Block(
                    'block5', bottleneck, [(num_fm, num_fm//4, 2)] + [(num_fm, num_fm//4, 1)] * (block_depth-1)),
                resnet_utils.Block(
                    'block6', bottleneck, [(num_fm, num_fm//4, 2)] + [(num_fm, num_fm//4, 1)] * (block_depth-1)),
                resnet_utils.Block(
                    'block7', bottleneck, [(num_fm, num_fm//4, 2)] + [(num_fm, num_fm//4, 1)] * (block_depth-1)),
            ]
            if args.image_size == 512:
                blocks += [
                    resnet_utils.Block(
                        'block8', bottleneck, [(num_fm, num_fm//4, 2)] + [(num_fm, num_fm//4, 1)] * (block_depth-1)),
                ]

            net, endpoints = resnet_v1.resnet_v1(net, blocks,
                                                 global_pool=False,
                                                 include_root_block=False,
                                                 reuse=self.reuse,
                                                 scope=DEFAULT_SSD_SCOPE)
            self.outputs.update(endpoints)
            with tf.variable_scope(DEFAULT_SSD_SCOPE+"_back", reuse=self.reuse):
                end_points_collection = "reverse_ssd_end_points"
                with slim.arg_scope([slim.conv2d, bottleneck_skip],
                                    outputs_collections=end_points_collection):
                    top_fm = args.top_fm
                    int_fm = top_fm//4
                    if args.image_size == 512:
                        # as long as the number of pooling layers is bigger due to
                        # the higher resolution, an extra layer is appended
                        net = bottleneck_skip(net, self.outputs[DEFAULT_SSD_SCOPE+'/block7'],
                                              top_fm, int_fm, scope='block_rev7')

                    net = bottleneck_skip(net, self.outputs[DEFAULT_SSD_SCOPE+'/block6'],
                                          top_fm, int_fm, scope='block_rev6')
                    net = bottleneck_skip(net, self.outputs[DEFAULT_SSD_SCOPE+'/block5'],
                                          top_fm, int_fm, scope='block_rev5')
                    net = bottleneck_skip(net, self.outputs[self.scope+'/block4'],
                                          top_fm, int_fm, scope='block_rev4')
                    net = bottleneck_skip(net, self.outputs[self.scope+'/block3'],
                                          top_fm, int_fm, scope='block_rev3')
                    net = bottleneck_skip(net, self.outputs[self.scope+'/block2'],
                                          top_fm, int_fm, scope='block_rev2')
                    if args.x4:
                        # To provide stride 4 we add one more layer with upsampling
                        net = bottleneck_skip(net, self.outputs[self.scope+'/block1'],
                                              top_fm, int_fm, scope='block_rev1')
                endpoints = slim.utils.convert_collection_to_dict(end_points_collection)
            self.outputs.update(endpoints)

            # Creating an output of spatial resolution 1x1 with conventional name 'pool6'
            if args.image_size == 512:
                self.outputs[DEFAULT_SSD_SCOPE+'/pool6'] =\
                        tf.reduce_mean(self.outputs['ssd_back/block_rev7/shortcut'],
                                       [1, 2], name='pool6', keep_dims=True)
            else:
                self.outputs[DEFAULT_SSD_SCOPE+'/pool6'] =\
                        tf.reduce_mean(self.outputs['ssd_back/block_rev6/shortcut'],
                                       [1, 2], name='pool6', keep_dims=True)

    def create_multibox_head(self, num_classes):
        """
        Creates outputs for classification and localization of all candidate bboxes
        """
        locations = []
        confidences = []
        with tf.variable_scope(DEFAULT_SSD_SCOPE, reuse=self.reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope(self.vgg_arg_scope()):
                with slim.arg_scope([slim.conv2d], outputs_collections=end_points_collection,
                                    weights_initializer=slim.variance_scaling_initializer(factor=0.1),
                                    activation_fn=None):
                    for i, layer_name in enumerate(self.layers):

                        if i > 0 and args.head == 'shared':
                            sc.reuse_variables()
                        if args.head == 'shared':
                            scope_suffix = ''
                        elif args.head == 'nonshared':
                            scope_suffix = '/'+layer_name
                        else:
                            raise ValueError
                        src_layer = self.outputs[layer_name]
                        shape = src_layer.get_shape()
                        w, h = shape[1], shape[2]
                        wh = shape[1] * shape[2]
                        batch_size = shape[0]
                        num_priors = len(self.config['aspect_ratios'][i])*2 + 2

                        loc = slim.conv2d(src_layer, num_priors * 4,
                                          [args.det_kernel, args.det_kernel],
                                          scope='location'+scope_suffix)
                        loc_sh = tf.stack([batch_size, wh * num_priors, 4])
                        locations.append(tf.reshape(loc, loc_sh))
                        tf.summary.histogram("location/"+layer_name, locations[-1])

                        conf = slim.conv2d(src_layer, num_priors * num_classes,
                                           [args.det_kernel, args.det_kernel],
                                           scope='confidence'+scope_suffix)
                        conf_sh = tf.stack([batch_size, wh * num_priors, num_classes])
                        confidences.append(tf.reshape(conf, conf_sh))
                        tf.summary.histogram("confidence/"+layer_name, confidences[-1])

                    ssd_end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                    self.outputs.update(ssd_end_points)
        all_confidences = tf.concat(confidences, 1)  # [layer_num, batch_size, wh * num_priors, num_classes]
        all_locations = tf.concat(locations, 1)
        # k = args.top_k_confidences
        # # top_confidences shape [layer_num, batch_size, wh * num_priors]
        # top_confidences, top_inds = tf.nn.top_k(all_confidences, 1)
        # # top_confidences shape [batch_size, layers_num * wh * num_priors]
        # top_confidences = tf.reshape(top_confidences, (batch_size, -1))
        # # [batch_size, k(100)] the top k scores and indices
        # top_k_confidences, top_k_inds = tf.nn.top_k(top_confidences, k)
        # # need to eliminate all aspect_ratios (all the aspects num are 6)
        # top_k_confidences = top_confidences // 6
        # top_k_inds = top_k_inds // 6
        # # top_k_confidences = tf.reshape(top_k_confidences, (-1,))
        # top_k_inds = tf.reshape(top_k_inds, (-1,))
        #
        # # eliminate same aspect_ratio
        # # top_k_inds, idx = tf.unique(top_k_inds)
        # # top_k_confidences = tf.gather(top_k_confidences, idx)
        #
        # # top_k_inds = tf.reshape(top_k_inds, (batch_size, -1))
        # # top_k_inds, _ = tf.nn.top_k(top_k_inds, k)
        # # top_k_inds = tf.reshape(top_k_inds, (-1,))
        #
        # roi_info = self.roi_bounds.cal_roi_info()
        # # shape [batch*num_rois (layer y1 x1 y2 x2)]
        # top_k_rois = tf.gather(roi_info, top_k_inds)
        # self.top_k_inds = tf.reshape(top_k_inds, (batch_size, k)) * 6
        # top_k_rois = tf.reshape(top_k_rois,
        #                         (batch_size, k, 5))  # tf.shape(top_k_rois)[1:]))
        # self.top_k_rois = top_k_rois
        # # top_k_arrs = self.roi_bounds.get_from_arrs(top_k_inds, top_k_confidences)
        #
        # # top_k_inds_perbatch = []
        # # # for each batch
        # # for i in range(batch_size):
        # #     confidence_i = biggest_confidence[:, i]
        # #     layer_num, wh_mul_numPriors = tf.shape(confidence_i)
        # #     # the shape of top_indices: k, top k from [layer_num * wh * num_priors]
        # #     top_values, top_indices = tf.nn.top_k(tf.reshape(confidence_i, (-1,)), k)
        # #     self.roi_bounds.get_from_arrs(top_indices, top_values)
        # #     top_k_inds_perbatch.append(top_indices)

        self.outputs['location'] = all_locations
        self.outputs['confidence'] = all_confidences
        return all_confidences, all_locations

    def create_segmentation_head(self, num_classes):
        """segmentation of map with stride 8 or 4, if --x4 flag is active"""
        with tf.variable_scope(DEFAULT_SSD_SCOPE) as sc:
            with slim.arg_scope([slim.conv2d],
                                kernel_size=args.seg_filter_size,
                                weights_regularizer=slim.l2_regularizer(self.weight_decay),
                                biases_initializer=tf.zeros_initializer()):

                seg_materials = []
                seg_size = self.config['fm_sizes'][0]
                for i in range(len(self.layers)):
                    target_layer = self.outputs[self.layers[i]]
                    seg = slim.conv2d(target_layer, args.n_base_channels)
                    seg = tf.image.resize_nearest_neighbor(seg, [seg_size, seg_size])
                    seg_materials.append(seg)
                seg_materials = tf.concat(seg_materials, -1)
                seg_logits = slim.conv2d(seg_materials, num_classes,
                                         kernel_size=3, activation_fn=None)
                self.outputs['segmentation'] = seg_logits
                return self.outputs['segmentation']

    def create_instance_head(self, num_classes, rois, train_bn=True):
        """
        instance segmentation mask head
        Params
        ROIs: [batch, num_rois, H, W, C], the C of the initial Rois may not identical
        so it needs a preprocess first.
        """
        # test_x = tf.random_uniform((16, 700, 3, 3, 512))
        # print('test x shape: ', test_x.shape, tf.shape(test_x))
        # print('test x: ', test_x)
        # print('x shape: ', x.shape, tf.shape(x))
        # print('x: ', x)
        # t1 = KL.Conv2D(256, (3, 3), padding="same")(x[:, 0, :, :, :])
        # print(t1)
        # t2 = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
        #                          name="test_mask_test")(x)
        # print(t2)
        # t_x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
        #                          name="test_mask_conv1")(test_x)

        with tf.variable_scope(DEFAULT_SSD_SCOPE) as sc:
            feature_maps = []
            for i in range(len(self.layers)):
                feature_maps.append(self.outputs[self.layers[i]])
            x = PyramidROIExtract([args.det_kernel, args.det_kernel],
                                  self.config,
                                  name="roi_align_mask")([rois] + feature_maps)

            x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                                   name="instance_mask_deconv1")(x[:, :, :, :, :])
            # Conv layers
            x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                                   name="instance_mask_conv1")(x)
            x = KL.TimeDistributed(BatchNorm(),
                                   name='instance_mask_bn1')(x, training=train_bn)
            x = KL.Activation('relu')(x)

            x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                                   name="instance_mask_conv2")(x)
            x = KL.TimeDistributed(BatchNorm(),
                                   name='instance_mask_bn2')(x, training=train_bn)
            x = KL.Activation('relu')(x)

            x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                                   name="instance_mask_deconv2")(x)
            x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                                   name="instance_mask_conv3")(x)
            x = KL.TimeDistributed(BatchNorm(),
                                   name='instance_mask_bn3')(x, training=train_bn)
            x = KL.Activation('relu')(x)

            x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                                   name="instance_mask_conv4")(x)
            x = KL.TimeDistributed(BatchNorm(),
                                   name='instance_mask_bn4')(x, training=train_bn)
            x = KL.Activation('relu')(x)

            x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                                   name="instance_mask_deconv2")(x)
            x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                                   name="instance_mask")(x)
            return x

    def get_imagenet_init(self, opt):
        """optimizer is useful to extract slots corresponding to Adam or Momentum
        and exclude them from checkpoint assigning"""
        variables = slim.get_variables_to_restore(include=[self.scope])
        slots = set()
        for v in tf.trainable_variables():
            for s in opt.get_slot_names():
                slot = opt.get_slot(v, s)
                if slot is not None:
                    slots.add(slot)
        variables = list(set(variables) - slots)
        return slim.assign_from_checkpoint(self.ckpt, variables) + (variables, )


class ROI_bounds(object):
    def __init__(self, config):
        self.config = config
        self.fm_sizes = self.config["fm_sizes"]
        self.roi_bound = [0]
        self.num_priors = []

        for i, fm_size in enumerate(self.fm_sizes):
            num_prior = len(self.config['aspect_ratios'][i])*2 + 2
            self.num_priors.append(num_prior)
            roi_bound_i = self.roi_bound[-1] + fm_size ** 2 * num_prior
            self.roi_bound.append(roi_bound_i)

    def get_roi_feature_pos(self, roi_idx, select='layer'):
        """give the idx of the roi, output the [layer, w, h] of roi"""
        roi_bound = self.roi_bound
        for i in range(len(roi_bound)-1):
            if roi_idx >= roi_bound[i] and roi_idx < roi_bound[i+1]:
                layer_num = i
                rel_roi_idx = roi_idx - roi_bound[i]
                rel_roi_idx = rel_roi_idx // self.num_priors[i]
                w = rel_roi_idx // self.fm_sizes[i]
                h = rel_roi_idx % self.fm_sizes[i]

        return filte_output(tf.constant([layer_num]), w, h, select)

    def get_from_arrs(self, roi_idxs, top_values):
        with tf.Session() as sess:
            roi_idx_arr = roi_idxs.eval()
            top_values_arr = top_values.eval()
        roi_arr = np.concatenate(list(map(get_roi_feature_pos,
                                          roi_idx_arr))).reshape(-1, 3)
        roi_arr = np.concatenate([roi_arr, top_values_arr.reshape(-1, 1)], axis=1)
        return roi_arr

    def filte_output(layer_num, w, h, select='layer'):
        if select == 'layer':
            return layer_num
        if select == 'w':
            return w
        if select == 'h':
            return h

    def cal_roi_info(self):
        roi_info = []
        # roi in shape [layer_num, y1, x1, y2, x2] normalized
        # (needs to take the aspect_ratios into account)
        for layer_num, fm_size in enumerate(self.fm_sizes):
            # num_prior = len(self.config['aspect_ratios'][layer_num])*2 + 2
            for w in range(fm_size):
                for h in range(fm_size):
                    radius = args.det_kernel // 2
                    y1 = (h - radius) / fm_size
                    x1 = (w - radius) / fm_size
                    y2 = (h + radius) / fm_size
                    x2 = (w + radius) / fm_size
                    roi_per_prior = [[layer_num, y1, x1, y2, x2]]  # * num_prior
                    roi_info.extend(roi_per_prior)

        self.roi_info = tf.stack(roi_info, 0)
        return self.roi_info

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


class PyramidROIExtract(KE.Layer):
    """Implements ROI extracting on multiple layers of the SSD

    Params:
    - output shape: [output_h, output_w] of the extracted regions. defalut [3, 3]

    Inputs:
    - feature pos: [batch, num_rois, (layer_num, y1, x1, y2, x2)]
    - feature maps: the list of all SSD rev feature maps

    Output:
    extracted regions in the shape: [batch, num_rois, output_h, output_w, channels]
    """

    def __init__(self, roi_shape, config, **kwargs):
        super(PyramidROIExtract, self).__init__(**kwargs)
        self.roi_shape = tuple(roi_shape)
        self.num_rois = args.instance_num
        self.config = config
        self.roi_layer_num = len(self.config['fm_sizes'])

    def call(self, inputs):
        roi_pos = inputs[0]
        feature_maps = inputs[1:]

        layer_num, y1, x1, y2, x2 = tf.split(roi_pos, 5, axis=2)
        h = y2 - y1
        w = x2 - x1
        roi_pos = roi_pos[:, :, 1:]

        rois = []
        roi_to_level = []
        layer_num = tf.squeeze(layer_num, 2)
        for i, level in enumerate(range(self.roi_layer_num)):
            ix = tf.where(tf.equal(layer_num, level))
            level_rois = tf.gather_nd(roi_pos, ix)

            # Box indices for crop_and_resize.
            roi_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            roi_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_rois = tf.stop_gradient(level_rois)
            roi_indices = tf.stop_gradient(roi_indices)
            # Result: [batch * num_rois, output_h, output_w, channels]
            # p.s. the batch infomation is lost when using gather_nd to
            # extract the target level
            rois.append(tf.image.crop_and_resize(
                feature_maps[i], level_rois, roi_indices, self.roi_shape,
                method="bilinear"))

        # Pack rois features into one tensor
        rois = tf.concat(rois, axis=0)

        # Pack roi_to_level mapping into one array add another
        # column representing the order of roi boxes
        roi_to_level = tf.concat(roi_to_level, axis=0)
        roi_range = tf.expand_dims(tf.range(tf.shape(roi_to_level)[0]), 1)
        roi_to_level = tf.concat([tf.cast(roi_to_level, tf.int32), roi_range],
                                 axis=1)

        # Rearrange selected features to match the order of the original boxes
        # Sort roi_to_level by batch then roi index
        # TF dosen't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = roi_to_level[:, 0] * 100000 + roi_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            roi_to_level)[0]).indices[::-1]
        ix = tf.gather(roi_to_level[:, 2], ix)
        rois = tf.gather(rois, ix)

        rois_shape = tf.concat([tf.shape(roi_pos)[:1],  # batch
                                tf.reshape(tf.constant(self.num_rois), (1,)),  # -1
                                tf.shape(rois)[1:]], axis=0)  # output_h, output_w, channels
        rois = tf.reshape(rois, rois_shape)

        return rois
