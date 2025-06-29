"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from models import Res
from sklearn.neighbors import KNeighborsClassifier


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAR(nn.Module):
    """SAR online adapts a net by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a net adapts itself by updating on every forward.
    """
    def __init__(self, net, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.imagenet_mask=None
        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for net recovery scheme
        self.ema = None  # to record the moving average of net output entropy, as net recovery criteria

        # note: if the net is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.net_state, self.optimizer_state = \
            copy_net_and_optimizer(self.net, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag = forward_and_adapt_sar(x, self.net, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema, self.imagenet_mask)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        return outputs

    def reset(self):
        if self.net_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved net/optimizer state")
        load_net_and_optimizer(self.net, self.optimizer,
                                 self.net_state, self.optimizer_state)
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, net, optimizer, margin, reset_constant, ema, imagenet_mask):
    """Forward and adapt net input data.
    Measure entropy of the net prediction, take gradients, and update params.
    """
#    inputs,targets,loss_fct,net,defined_backward = self.paras
    #assert defined_backward is not None, "Sharpness Aware Minimization requires defined_backward, but it was not provided"

    optimizer.zero_grad()
    # forward
    outputs = net(x)
    if imagenet_mask is not None:
        outputs = outputs[:, imagenet_mask]
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    entropys = softmax_entropy(outputs)
    
    filter_ids_1 = torch.where(entropys < margin)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()
    optimizer.first_step(zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
    
    outputs2=net(x)
    if imagenet_mask is not None:
        outputs2 = outputs2[:, imagenet_mask]
    entropys2 = softmax_entropy(outputs2)
    # second time forward  
    
    filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since net weights have been changed to \Theta+\hat{\epsilon(\Theta)}
    loss_second = entropys2[filter_ids_2].mean(0)
    loss_second_value = loss_second.clone().detach().mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  # record moving average loss values for net recovery

    # second time backward, update net weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    loss_second.backward()
    optimizer.second_step(zero_grad=True)
    # perform net recovery
    reset_flag = False
    if ema is not None:
        if ema < 0.2:
            print("ema < 0.2, now reset the net")
            reset_flag = True
    return outputs, ema, reset_flag


def collect_params(net):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the net's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in net.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_net_and_optimizer(net, optimizer):
    """Copy the net and optimizer states for resetting after adaptation."""
    net_state = deepcopy(net.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return net_state, optimizer_state


def load_net_and_optimizer(net, optimizer, net_state, optimizer_state):
    """Restore the net and optimizer states from copies."""
    net.load_state_dict(net_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(net):
    """Configure net for use with SAR."""
    # train mode, because SAR optimizes the net to minimize entropy
    net.train()
    # disable grad, to (re-)enable only what SAR updates
    net.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN nets)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN nets
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return net


def check_net(net):
    """Check net for compatability with SAR."""
    is_training = net.training
    assert is_training, "SAR needs train mode: call net.train()"
    param_grads = [p.requires_grad for p in net.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SAR needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "SAR should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in net.modules()])
    assert has_norm, "SAR needs normalization layer parameters for its optimization"
