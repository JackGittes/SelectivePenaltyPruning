"""
Author: Zhao Mingxin
Date:   2020/07/08
Description: Criteria for choosing weak channels.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import scipy.io as sio
import time
import random


def output_oriented_importance(model, feature_root, prune_ratio, start_point, display=True, criterion='ThiNet'):
    """
    Initialize all theta params of conv1 output.
    TODO: how to fuse theta(gating) operation with batch normalization layer? This may do help to improve
    TODO: training performance and speed.
    :param model: resnet model
    :param feature_root:  omitted
    :param prune_ratio:  omitted
    :param start_point:  the start point for weak channels to begin their competition.
    :param display: print channel selection progress or not.
    :param criterion: specify which method is employed to select weak channels, default is ThiNet.
    :return:
    """
    _method = ['ThiNet', 'APoZ', 'NormMag', 'WeightSum', 'Random']

    assert criterion in _method, "Unknown weak channel selection method."

    feature_files = os.listdir(feature_root)
    chosen_blocks = []
    for _item in feature_files:
        if _item[-4:] == '.mat' and 'conv2' in _item:
            chosen_blocks.append(_item)
    _begin_time = time.time()
    for _block_file in chosen_blocks:
        _start_time = time.time()
        if display:
            print("Start selection of layer: {} \n".format(_block_file[:-4]))

        _block_features = sio.loadmat(os.path.join(feature_root, _block_file))
        _in_features = torch.from_numpy(_block_features['in'])
        name_parts = _block_file.split('.')
        _layer = getattr(getattr(model, str(name_parts[0]))[int(name_parts[1])], 'conv2')
        weight, groups, stride, padding = _layer.weight.data, _layer.groups, _layer.stride, _layer.padding

        if criterion == 'ThiNet':
            pruned_chn = _output_oriented_greedy_selection(weight, stride, groups, padding, _in_features, prune_ratio)
        elif criterion == 'NormMag':
            pruned_chn = norm_magnitude(_in_features, prune_ratio)
        elif criterion == 'WeightSum':
            c1_weight = getattr(getattr(model, str(name_parts[0]))[int(name_parts[1])], 'conv1').weight
            pruned_chn = weight_sum_criterion(c1_weight, prune_ratio)
        elif criterion == 'APoZ':
            pruned_chn = apoz_selection(_in_features, prune_ratio)
        elif criterion == 'Random':
            pruned_chn = random_init_channel_scores(_in_features, prune_ratio)
        else:
            raise RuntimeError("Unknown error.")

        _init_layer = getattr(model, str(name_parts[0]))[int(name_parts[1])]
        orig_theta = torch.ones(_init_layer.conv1.weight.size(0))
        for idx in pruned_chn:
                orig_theta[idx] = start_point
        _init_layer.theta_1 = nn.Parameter(orig_theta, requires_grad=True)

        _end_time = time.time()
        if display:
            print("End of channel selection for layer: {} \n".format(_block_file[:-4]))
            print(" = Time: {:>3.2f} s".format(_end_time - _start_time))
    _end_time = time.time()
    if display:
        print("{} method channel selection time: {:>3.2f} s".format(criterion, _end_time - _begin_time))


def _output_oriented_greedy_selection(weight, stride, groups, padding, in_features, prune_ratio):
    """
    Re-implemented greedy channel selection algorithm proposed in ThiNet.
    Paper Ref:  Luo, Jian-Hao, Jianxin Wu, and Weiyao Lin. "Thinet: A filter level pruning method for deep neural
                network compression." Proceedings of the IEEE international conference on computer vision. 2017.
    :param weight: convolution weight.
    :param stride: stride for convolution.
    :param groups: omitted.
    :param padding: omitted.
    :param in_features: input features of current layer.
    :param prune_ratio: required_pruned_channels / total_input_channels.
    :return: the indexes of channels that needed to be pruned.
    """
    assert isinstance(in_features, torch.Tensor)
    assert isinstance(prune_ratio, float) and prune_ratio < 1.0, "prune ratio must be a floating-point number " \
                                                                 "and less than 1.0."
    _in_chn = in_features.size(1)
    required_pruned = int(np.ceil(prune_ratio * _in_chn))
    temp_pruned, pruned, original_chn = set(), set(), set(range(_in_chn))

    while len(pruned) < required_pruned:
        min_value, min_i = torch.tensor(float("Inf")), None
        for _i in original_chn:
            temp_pruned = pruned.union({_i})
            temp_mask = torch.zeros(_in_chn)
            for _masked in temp_pruned:
                temp_mask[_masked] = 1
            with torch.no_grad():
                masked_weight = temp_mask[None, :, None, None] * weight
                res = func.conv2d(in_features, masked_weight, groups=groups, padding=padding, stride=stride)
                value = torch.norm(res, p=2)
            if value < min_value:
                min_value, min_i = value, _i
        assert min_i is not None, "min_i is invalid, it should not be None type."
        pruned.add(min_i)
        original_chn.remove(min_i)
    return list(pruned)


def apoz_selection(_tensor, prune_ratio):
    out_chn = _tensor.size(1)
    required_chn_num = int(np.ceil(out_chn * prune_ratio))
    scores = average_percentage_of_zeros(_tensor)
    pruned = list(np.argsort(np.asarray(scores))[:required_chn_num])
    return pruned


def average_percentage_of_zeros(_tensor):
    """
    Re-implemented APoZ criterion for selecting potential weak channels.
    :param _tensor: activations of a convolution layer output.
    :return: scores of different channels corresponding to its APoZ.
    """
    assert isinstance(_tensor, torch.Tensor)
    out_chn = _tensor.size(1)
    scores = []
    for _chn in range(out_chn):
        sub_chn_tensor = _tensor[:, _chn, :, :]
        chn_score = float(torch.sum(_tensor[:, _chn, :, :] == 0)) / float(torch.numel(sub_chn_tensor))
        scores.append(float(chn_score))
    return scores


def norm_magnitude(_tensor, prune_ratio):
    """
    :param _tensor: pre-activated convolution result.
    :param prune_ratio: required_pruned_channels / total_channels
    :return: selected channels.
    """
    assert isinstance(_tensor, torch.Tensor), "Input #arg 1 must be a tensor."
    activated_ = func.relu(_tensor)
    out_chn = _tensor.size(1)
    required_chn_num = int(np.ceil(out_chn * prune_ratio))
    norms_chn = []
    for _i in range(out_chn):
        norms_chn.append(float(torch.norm(activated_[:, _i, :, :], p=2) / _tensor.size(0)))
    norm_array = np.asarray(norms_chn)
    scores = norm_array / (np.max(norm_array) + 1e-9)

    pruned = list(np.argsort(scores)[:required_chn_num])
    return pruned


def weight_sum_criterion(_weight, prune_ratio):
    """
    Select weak channels according to their weights' L1-norm.
    :param _weight:
    :param prune_ratio:
    :return:
    """
    assert isinstance(_weight, torch.Tensor), "Input arg #1 must be a tensor."
    out_chn = _weight.size(0)
    required_chn_num = int(np.ceil(out_chn * prune_ratio))
    scores = []
    for _i in range(out_chn):
        scores.append(float(torch.norm(_weight[_i, :, :, :], p=1)))
    score_array = np.asarray(scores)
    pruned = list(np.argsort(score_array)[:required_chn_num])
    return pruned


def random_init_channel_scores(_tensor, prune_ratio):
    """
    Randomly select channels for pruning.
    :param _tensor: output tensor of required pruned layer.
    :param prune_ratio: [0, 1.0], a floating-point number.
    :return:
    """
    assert isinstance(_tensor, torch.Tensor), "Input arg #1 must be a tensor."
    out_chn = _tensor.size(1)
    required_chn_num = int(np.ceil(out_chn * prune_ratio))
    chn_list = list(range(out_chn))
    random.shuffle(chn_list)
    pruned = chn_list[:required_chn_num]
    return pruned
