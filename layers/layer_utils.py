from collections.abc import Iterable
import os.path as osp
import os
import warnings
from collections import namedtuple
from typing import List, Tuple, Optional
import numbers
import math

#for quantization
import torch.utils.data
import pytorch_quantization.utils
import pytorch_quantization.nn.modules._utils as _utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn
#


def _ntuple(n):
    '''Function for handling scalar and iterable layer arguments'''

    def parse(x):
        '''Closure for parsing layer args'''
        if isinstance(x, Iterable):
            return x
        return tuple([x for i in range(n)])

    return parse


# Typedef
_pair = _ntuple(2)