import math
import torch
from torch import Tensor
from typing import List, Optional
import numpy as np


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        if i == 0:
            print("before")
            print(d_p)
            print("param")
            print(param)
            print("transpose")
            transpose = torch.transpose(param, 0, 1)
            print(transpose)
            print("multiplication")
            multiplication = torch.matmul(transpose, param)
            print(multiplication)
            print("diag")
            npI = np.identity(multiplication.size()[0])
            npI = npI.astype("float32")
            I = torch.from_numpy(npI)
            print(I)
            print("t")
            t = torch.sub(I, multiplication)
            print(t)
            d_p = torch.matmul(d_p, t)
            print("after")
            print(d_p)

        param.add_(d_p, alpha=-lr)
