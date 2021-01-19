"""
Author: Zhao Mingxin
Date:   2020/07/13
Description: Sparsity estimation for Mobilenet-V1 on ILSVRC-2012.
"""

import torch
import torch.nn as nn


def normal_conv2d_flops(k_w, in_chn, out_chn, o_w, o_h=None):
    if o_h is None:
        o_h = o_w
    # TODO: To be consistent with other papers in pruning field, the flops calculation
    # TODO: ONLY accounts for multiply-add operations without regarding pure addition operations.
    # (((ker_w ** 2) + (ker_w ** 2 - 1)) * channel_ + channel_ - 1) * o_w * o_h
    return ((k_w**2)*in_chn + in_chn - 1)*o_w*o_h*out_chn


def conv2d_params(k_w, in_chn, out_chn):
    return k_w ** 2 * in_chn * out_chn


def depth_sep_conv2d_flops(k_w, in_chn, out_chn, o_w, o_h=None):
    if o_h is None:
        o_h = o_w
    return ((k_w ** 2)*in_chn + (in_chn - 1)*out_chn)*o_w*o_h


def depth_sep_conv2d_param(k_w, in_chn, out_chn):
    depthwise_param = k_w ** 2 * in_chn
    normal_conv_param = in_chn * out_chn
    return depthwise_param + normal_conv_param


def compute_sparsity(net, trap_thresh, masked, display=False, parallel=False):

    if parallel:
        model = net.module.features
    else:
        model = net.features

    total_param = 0.0
    pruned_param = 0.0
    report_str = ''

    total_param += depth_sep_conv2d_param(3, 32, 16)

    if masked:
        now_in_chn = float(torch.sum(model[1].conv[0][1].mask))
        now_out_chn = float(torch.sum(model[1].conv[2].mask))
    else:
        now_in_chn = float(torch.sum(model[1].conv[0][1].theta > trap_thresh))
        now_out_chn = float(torch.sum(model[1].conv[2].theta > trap_thresh))
    layer_pruned = depth_sep_conv2d_param(3, 32, 16) - \
                   depth_sep_conv2d_param(3, now_in_chn, now_out_chn)
    pruned_param += layer_pruned

    report_str += 'N#: features.{:>2}.depsep  S#: [{:>4}, {:>4}], P#: [{:>4}, {:>4}], ' \
                  'SP#: {:>5}\n'.format(1, int(32), int(16),
                                        int(now_in_chn), int(now_out_chn), int(layer_pruned))

    for _i in range(2, 18):
        now_in_chn = now_out_chn
        prev_in_chn, prev_out_chn = model[_i].conv[0][0].in_channels, model[_i].conv[0][0].out_channels
        if masked:
            now_out_chn = float(torch.sum(model[_i].conv[0][1].mask))
        else:
            now_out_chn = float(torch.sum(model[_i].conv[0][1].theta > trap_thresh))
        layer_pruned = conv2d_params(1, prev_in_chn, prev_out_chn) - conv2d_params(1, now_in_chn, now_out_chn)
        total_param += float(torch.numel(model[_i].conv[0][0].weight))
        pruned_param += layer_pruned

        report_str += 'N#: features.{:>2}.conv2d  S#: [{:>4}, {:>4}], P#: [{:>4}, {:>4}], ' \
                      'SP#: {:>5}\n'.format(_i, int(prev_in_chn), int(prev_out_chn),
                                            int(now_in_chn), int(now_out_chn), int(layer_pruned))

        now_in_chn = now_out_chn
        prev_in_chn = prev_out_chn

        if not isinstance(model[_i].conv[3], nn.BatchNorm2d):
            prev_out_chn = float(torch.numel(model[_i].conv[3].mask))
            if masked:
                now_out_chn = float(torch.sum(model[_i].conv[3].mask))
            else:
                now_out_chn = float(torch.sum(model[_i].conv[3].theta > trap_thresh))
        else:
            prev_out_chn = float(torch.numel(model[_i].conv[3].weight))

            if masked:
                now_out_chn = float(torch.sum(model[_i].mask))
            else:
                now_out_chn = float(torch.sum(model[_i].theta > trap_thresh))

        layer_pruned = depth_sep_conv2d_param(3, prev_in_chn, prev_out_chn) - depth_sep_conv2d_param(3, now_in_chn, now_out_chn)
        total_param += depth_sep_conv2d_param(3, prev_in_chn, prev_out_chn)
        pruned_param += layer_pruned

        report_str += 'N#: features.{:>2}.depsep  S#: [{:>4}, {:>4}], P#: [{:>4}, {:>4}], ' \
                      'SP#: {:>5}\n'.format(_i, int(prev_in_chn), int(prev_out_chn),
                                            int(now_in_chn), int(now_out_chn), int(layer_pruned))

    now_in_chn = now_out_chn
    if masked:
        now_out_chn = float(torch.sum(model[18][1].mask))
    else:
        now_out_chn = float(torch.sum(model[18][1].theta > trap_thresh))
    total_param += float(torch.numel(model[18][0].weight))
    layer_pruned = conv2d_params(1, 320, 1280) - conv2d_params(1, now_in_chn, now_out_chn)
    pruned_param += layer_pruned

    report_str += 'N#: features.{:>2}.depsep  S#: [{:>4}, {:>4}], P#: [{:>4}, {:>4}], ' \
                  'SP#: {:>5}\n'.format(18, int(320), int(1280),
                                        int(now_in_chn), int(now_out_chn), int(layer_pruned))

    total_param += 1280 * 1000
    pruned_param += 1280 * 1000 - now_out_chn * 1000

    report_str += 'N#: features.{:>2}.linear  S#: [{:>4}, {:>4}], P#: [{:>4}, {:>4}], ' \
                  'SP#: {:>5}\n'.format(18, int(1280), int(1000),
                                        int(now_out_chn), int(1000), int(1280 * 1000 - now_out_chn * 1000))

    sparsity_ratio = pruned_param / total_param

    if display:
        print("============== Sparsity ratio estimation ==============\n")
        print(report_str)
        print("Total parameters: {}".format(int(total_param)))
        print("Pruned parameters: {}".format(int(pruned_param)))
        print("Sparsity ratio: {:>2.2f} %\n".format(sparsity_ratio * 100))
        print("=======================================================\n")
    return sparsity_ratio
