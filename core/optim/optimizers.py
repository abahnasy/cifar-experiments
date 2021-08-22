from functools import partial
from typing import Any, Dict, List, Optional, Union


import torch.optim as optim
from torch.optim import adadelta, adagrad, adamax, rmsprop, rprop
from torch.optim.optimizer import Optimizer
from ranger import Ranger
from ranger21 import Ranger21

from omegaconf import DictConfig, OmegaConf

AVAILABLE_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'adadelta': adadelta.Adadelta,
    'adamax': adamax.Adamax,
    'adagrad': adagrad.Adagrad,
    'rmsprop': rmsprop.RMSprop,
    'rprop': rprop.Rprop,
    'ranger': Ranger,
    'ranger21': Ranger21
}

__all__ = [' parse_optimizer_args', 'get_optimizer']

def  parse_optimizer_args(
    cfg: Union[DictConfig, Dict[str, Any]]
)-> Union[Dict[str, Any], DictConfig]:
    """
    """

    # TODO: Filter arguments and return dictionary for initialization




def get_optimizer(name: str, **kwargs: Optional[Dict[str, Any]]) -> Optimizer:
    """
    Convenience method to obtain an Optimizer class and partially instantiate it with optimizer kwargs.
    Args:
        name: Name of the Optimizer in the registry.
        kwargs: Optional kwargs of the optimizer used during instantiation.
    Returns:
        a partially instantiated Optimizer
    """
    if name not in AVAILABLE_OPTIMIZERS:
        raise ValueError(
            f"Cannot resolve optimizer '{name}'. Available optimizers are : " f"{AVAILABLE_OPTIMIZERS.keys()}"
        )

    optimizer = AVAILABLE_OPTIMIZERS[name]
    optimizer = partial(optimizer, **kwargs)
    return optimizer