from paddle.io import Sampler
import paddle
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import numpy as np

class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        super(SubsetRandomSampler, self).__init__()
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)