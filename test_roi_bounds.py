import tensorflow as tf
import config
import numpy as np


class roi_bounds(object):
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
        # layer_num, w, h (needs to take the aspect_ratios into account)
        for layer_num, fm_size in enumerate(self.fm_sizes):
            num_prior = len(self.config['aspect_ratios'][layer_num])*2 + 2
            for w in range(fm_size):
                for h in range(fm_size):
                    roi_per_prior = [[layer_num, w, h]] * num_prior
                    roi_info.extend(roi_per_prior)

        self.roi_info = tf.stack(roi_info, 0)
