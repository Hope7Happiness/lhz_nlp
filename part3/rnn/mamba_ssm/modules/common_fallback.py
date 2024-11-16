import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numbers

def layer_norm_fn(
    x,
    weight,
    bias,
    residual,
    prenorm,
    residual_in_fp32,
    eps,
    is_rms_norm,
):
    assert is_rms_norm
    return rms_norm_fn(x, weight, bias, residual, eps, 0.0, prenorm, residual_in_fp32)
    assert bias is None
    print('x:',x.shape)
    print('weight:',weight.shape)
    print('bias:',None if bias is None else bias.shape)
    print('residual:',None if residual is None else residual.shape)
    print('prenorm:',prenorm)
    print('residual_in_fp32:',residual_in_fp32)
    print('eps:',eps)
    print('is_rms_norm:',is_rms_norm)
    raise NotImplementedError    
    
    
def rms_norm_fn(
    x: torch.Tensor,
    weight,
    bias,
    residual,
    eps: float,
    dropout_p=0.0,
    prenorm=False,
    residual_in_fp32=False,
    # normalized_shape: List[int],
    # weight:/ Optional[torch.Tensor] = None,
    # eps: float = 1e-5,
):
    # norm_ndim = len(normalized_shape)
    if residual is not None:
        x = x + residual
    xc = x
    norm_ndim = 1
    if False: # torch.jit.is_scripting():
        # ndim = len(x.shape)
        # dims = list(range(ndim - norm_ndim, ndim))  # this doesn't work on pytorch <= 1.13.x
        # NOTE -ve dims cause torchscript to crash in some cases, out of options to work around
        assert norm_ndim == 1
        v = torch.var(x, dim=-1).unsqueeze(-1)  # ts crashes with -ve dim + keepdim=True
    else:
        dims = tuple(range(-1, -norm_ndim - 1, -1))
        v = torch.var(x, dim=dims, keepdim=True)
    x = x * torch.rsqrt(v + eps)
    assert dropout_p < 0.001, f"dropout_p={dropout_p} not supported"
    if weight is not None:
        x = x * weight
    if bias is not None:
        x = x + bias
    return x if not prenorm else (x, xc)

class RMSNorm(nn.Module):
    """ RmsNorm w/ fast (apex) norm if available
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self,hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if dropout_p > 0.0:
            self.drop = torch.nn.Dropout(dropout_p)
        else:
            self.drop = None
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=self.drop.p if self.drop is not None and self.training else 0.0,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )
    
def silu(x):
    return x * F.sigmoid(x)

class SiLU(nn.Module):
    def forward(self, x):
        return silu(x)