from torchvision.models import mobilenet_v2
from prune.utils import replace_bn
from prune.statistic import compute_sparsity

net = mobilenet_v2()
replace_bn(net)

compute_sparsity(net, 0.0, False, True, False)
