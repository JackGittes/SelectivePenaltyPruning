from model.mobilenet import InvertedResidual
import torch
import torch.nn as nn


class PruneInvertedResidual(nn.Module):
    def __init__(self, inverted_residual):
        super(PruneInvertedResidual, self).__init__()
        assert isinstance(inverted_residual, InvertedResidual), "arg #1 must be an inverted residual module."
        self.conv = inverted_residual.conv
        self.use_res_connect = inverted_residual.use_res_connect

        self.out_p = torch.numel(self.conv[-1].weight)
        self.theta = nn.Parameter(torch.ones(self.out_p), requires_grad=True)
        self.mask = nn.Parameter(torch.ones(self.out_p), requires_grad=False)
        self.masked = None

    def forward(self, x):
        if self.use_res_connect:
            return (x + self.conv(x)) * self.theta[None, :, None, None] * self.mask[None, :, None, None]
        else:
            return self.conv(x) * self.theta[None, :, None, None] * self.mask[None, :, None, None]

    def mask_gradient(self):
        self.mask[torch.abs(self.theta) < self.thresh] = 0.0
        theta_data = self.theta.data
        theta_data[torch.abs(self.theta) < self.thresh] = 0.0
        self.theta.data = theta_data

        def backward_hook(grad):
            out = grad.clone()
            masked_grad = out * self.mask
            return masked_grad

        self.masked = True
        self.theta.register_hook(backward_hook)


class Penalty(nn.Module):
    def __init__(self, k):
        super(Penalty, self).__init__()
        assert isinstance(k, float) or isinstance(k, int), "k step must be a float or integer number."
        self.k = k

    def forward(self, x, k=1):
        self.k = k
        div = 1 + torch.exp(- self.k * torch.abs(x))
        res = torch.sum(torch.div(2, div) - 1)
        return res


class PruneBatchNorm(nn.Module):
    """
    Wrap batch normalization with weighting operation.
    """
    def __init__(self, bn):
        super(PruneBatchNorm, self).__init__()
        assert isinstance(bn, nn.BatchNorm2d), 'Input arg #1 must be a batch normalization layer.'
        self.bn = bn
        self.theta = nn.Parameter(torch.ones(torch.numel(self.bn.weight)), requires_grad=True)
        self.mask = nn.Parameter(torch.ones(torch.numel(self.bn.weight)), requires_grad=False)

        self.suppression_ratio = None
        self.thresh = None
        self.masked = False

    def forward(self, x):
        x = self.bn(x)
        x = x * self.theta[None, :, None, None]
        return x

    def mask_gradient(self):
        self.mask[torch.abs(self.theta) < self.thresh] = 0.0
        theta_data = self.theta.data
        theta_data[torch.abs(self.theta) < self.thresh] = 0.0
        self.theta.data = theta_data

        def backward_hook(grad):
            out = grad.clone()
            masked_grad = out * self.mask
            return masked_grad

        self.masked = True
        self.theta.register_hook(backward_hook)


def penalty_loss(net, alpha_0):
    p = Penalty(k=1)
    total_loss = 0.0
    for m_ in net.modules():
        if isinstance(m_, PruneBatchNorm) or isinstance(m_, PruneInvertedResidual):
            total_loss += p(m_.theta, alpha_0)
    return total_loss
