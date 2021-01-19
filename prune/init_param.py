import torch
from .layer import PruneBatchNorm, PruneInvertedResidual
from .criteria import norm_magnitude
import scipy.io as sio
import os


def init_model_theta(model, feature_root, start_point, prune_ratio):
    model.eval()
    with torch.no_grad():
        _layer = model.features[1].conv[0][1]
        init_layer(_layer, feature_root, 'features.1.bn1.mat', start_point, prune_ratio)

        _layer = model.features[1].conv[2]
        init_layer(_layer, feature_root, 'features.1.bn2.mat', start_point, prune_ratio)

        for _i in range(2, 18):
            _layer = model.features[_i].conv[0][1]
            init_layer(_layer, feature_root, 'features.{}.bn.mat'.format(_i), start_point, prune_ratio)

            if model.features[_i].use_res_connect:
                _layer = model.features[_i]
            else:
                _layer = model.features[_i].conv[3]
            init_layer(_layer, feature_root, 'features.{}.out.mat'.format(_i), start_point, prune_ratio)

        _layer = model.features[18][1]
        init_layer(_layer, feature_root, 'features.18.out.mat'.format(_i), start_point, prune_ratio)
    model.train()


def init_layer(layer, feature_root, features_mat_name, start_point, prune_ratio):
    assert isinstance(layer, PruneBatchNorm) or isinstance(layer, PruneInvertedResidual)
    features = sio.loadmat(os.path.join(feature_root, features_mat_name))['out']
    features = torch.from_numpy(features)
    pruned_ = norm_magnitude(features, prune_ratio)
    theta_ = torch.ones_like(layer.theta)
    for _idx in pruned_:
        theta_[_idx] = start_point
    layer.theta.data = theta_
