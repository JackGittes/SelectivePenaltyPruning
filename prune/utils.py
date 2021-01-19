import os
import scipy.io as sio
from .layer import PruneBatchNorm, PruneInvertedResidual


def replace_bn(net):
    _bn = PruneBatchNorm(net.features[1].conv[0][1])
    net.features[1].conv[0][1] = _bn
    _bn = PruneBatchNorm(net.features[1].conv[2])
    net.features[1].conv[2] = _bn

    for _i in range(2, 18):
        _layer = net.features[_i]
        _bn = _layer.conv[0][1]
        _wrapped_bn = PruneBatchNorm(_bn)
        _layer.conv[0][1] = _wrapped_bn

        if net.features[_i].use_res_connect:
            _wrapped_layer = PruneInvertedResidual(_layer)
        else:
            _bn = _layer.conv[3]
            _wrapped_bn = PruneBatchNorm(_bn)
            _layer.conv[3] = _wrapped_bn
            _wrapped_layer = _layer
        net.features[_i] = _wrapped_layer
    _bn = PruneBatchNorm(net.features[18][1])
    net.features[18][1] = _bn


def save_hook(save_name, save_folder):
    """
    This is the hook used to save the input and output tensors of a conv2d layer to a matlab file.
    :param save_name: name of the saved file.
    :param save_folder: folder to save the file.
    :return: real hook is returned to used by the registered module.
    """
    def real_hook(module, _, output_tensor):
        out_array = output_tensor.detach().cpu().numpy()
        # assert isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6) or \
        #     isinstance(module, nn.BatchNorm2d), "Not a relu module."
        save_dict = dict()
        save_dict['out'] = out_array
        sio.savemat(os.path.join(save_folder, save_name+'.mat'), save_dict)
    return real_hook


def hook_register_mobilenet_v2(net, save_folder):
    """
    This function register a forward hook for every fused conv2d layer to obtain its input and output.
    The recorded tensor is saved to the save_folder when forwarding the net.
    :param net: network needed to be hooked.
    :param save_folder: folder to save tensors.
    :return: hooked net.
    """
    net.features[1].conv[0][2].register_forward_hook(save_hook('features.1.bn1', save_folder=save_folder))
    net.features[1].conv[2].register_forward_hook(save_hook('features.1.bn2', save_folder=save_folder))

    for _i in range(2, 18):
        net.features[_i].conv[0][2].register_forward_hook(save_hook('features.{}.bn'.format(_i), save_folder=save_folder))

        if net.features[_i].use_res_connect:
            net.features[_i].register_forward_hook(save_hook('features.{}.out'.format(_i),
                                                             save_folder=save_folder))
        else:
            net.features[_i].conv[3].register_forward_hook(save_hook('features.{}.out'.format(_i),
                                                                     save_folder=save_folder))
    net.features[18][2].register_forward_hook(save_hook('features.18.out', save_folder=save_folder))


def forward_save(model, x, save_folder='features'):
    model.eval()
    hook_register_mobilenet_v2(model, save_folder)
    model.forward(x)
