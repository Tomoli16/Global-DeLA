""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2021 Ross Wightman
"""
import logging
from itertools import islice
from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False

_logger = logging.getLogger(__name__)


def param_groups_weight_decay(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def create_optimizer_v2(
        model_or_params,
        opt: str = 'adamw',
        lr: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        **kwargs,
):
    """ Create an optimizer.

    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model_or_params (nn.Module): model containing parameters to optimize
        opt: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    if isinstance(model_or_params, nn.Module):
        # a model was passed in, extract parameters and add weight decays to appropriate layers
        no_weight_decay = {}
        if hasattr(model_or_params, 'no_weight_decay'):
            no_weight_decay = model_or_params.no_weight_decay()

        if weight_decay and filter_bias_and_bn:
            parameters = param_groups_weight_decay(model_or_params, weight_decay, no_weight_decay)
            weight_decay = 0.
        else:
            parameters = model_or_params.parameters()
    else:
        # iterable of parameters or param groups passed in
        parameters = model_or_params

    opt_lower = opt.lower()
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(weight_decay=weight_decay, **kwargs)
    if lr is not None:
        opt_args.setdefault('lr', lr)
        
    optimizer = optim.AdamW(parameters, **opt_args)

    return optimizer

def create_optimizer_v3(
    model_or_params,
    opt: str = 'adamw',
    paramwise_cfg: dict = None,
    lr: Optional[float] = None,
    weight_decay: float = 0.,
    momentum: float = 0.9,
    filter_bias_and_bn: bool = True,
    **kwargs,
):
    """
    Create an optimizer with optional per-submodule learning rates and weight decays.

    Args:
        model_or_params (nn.Module or iterable): the model or list of parameters
        opt: optimizer name, currently only 'adamw' supported
        paramwise_cfg: dict mapping submodule names to {'lr':..., 'weight_decay':...}
        lr: global base lr (used if paramwise_cfg is None)
        weight_decay: global weight decay (used if paramwise_cfg is None)
        momentum: momentum for optimizers that use it
        filter_bias_and_bn: if True, filter bias/BN params from weight decay when paramwise_cfg is None
        **kwargs: extra optimizer‐specific args (e.g. betas)
    """
    # 1) Build parameter groups
    if isinstance(model_or_params, nn.Module) and paramwise_cfg is not None:
        # override: use exactly the submodules in paramwise_cfg
        param_groups = []
        for name, cfg in paramwise_cfg.items():
            submod = getattr(model_or_params, name)
            pg = {
                'params': submod.parameters(),
                'lr': cfg.get('lr', lr),
                'weight_decay': cfg.get('weight_decay', weight_decay),
            }
            param_groups.append(pg)
        parameters = param_groups
        # disable global weight_decay/filter logic
        weight_decay = 0.
    else:
        # fallback to original v2 logic
        if isinstance(model_or_params, nn.Module):
            if filter_bias_and_bn and weight_decay:
                # param_groups_weight_decay soll zwei Gruppen erstellen:
                #  - mit weight_decay
                #  - ohne weight_decay für bias/BN
                no_decay = {}
                if hasattr(model_or_params, 'no_weight_decay'):
                    no_decay = model_or_params.no_weight_decay()
                parameters = param_groups_weight_decay(
                    model_or_params, weight_decay, no_decay
                )
                weight_decay = 0.
            else:
                parameters = model_or_params.parameters()
        else:
            # iterable of params or already‐built param groups
            parameters = model_or_params

    # 2) Build optimizer args
    opt_lower = opt.lower()
    if opt_lower not in ['adamw']:
        raise ValueError(f"Unsupported optimizer: {opt}")
    optim_args = dict(weight_decay=weight_decay, **kwargs)
    if lr is not None:
        optim_args.setdefault('lr', lr)

    # 3) Instantiate optimizer
    optimizer = optim.AdamW(parameters, **optim_args)
    return optimizer
