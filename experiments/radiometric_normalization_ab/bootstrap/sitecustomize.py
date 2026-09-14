"""Repeatable seeds for this experiment and its production subprocesses only."""
import random
random.seed(0)
try:
    import numpy
    numpy.random.seed(0)
    import torch
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
except ImportError:
    pass
