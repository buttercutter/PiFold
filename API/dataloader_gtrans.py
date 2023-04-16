import numpy as np
import torch
from typing import Iterator
from math import ceil


class LengthGrouper(torch.utils.data.RandomSampler):
    def __init__(self, *args, group_len: bool = False, batch_size: int = 1, delta: int = 10,  **kwargs):
        """ default arguments reverts to RandomSampler behavior """
        super().__init__(*args, **kwargs)
        self.group_len = group_len
        self.batch_size = batch_size
        self.delta = delta

    def __iter__(self) -> Iterator[int]:
        """ groups items by length, then shuffles"""
        if not self.group_len or self.batch_size <= 2:
            yield from super().__iter__()

        lens = np.array([len(d["seq"]) for d in self.data_source])
        if self.delta > 0:
            lens += np.random.randint(-abs(self.delta), +abs(self.delta), size=lens.shape)
        seq_idxs = np.argsort(lens)
        # FIXME: hypnopump@ if this does not work, consider accounting for number of workers here.
        r_last = self.batch_size * 2
        r_lead = ceil(seq_idxs.shape[0] / r_last)
        npad = r_lead * r_last - seq_idxs.shape[0]
        pad_idxs = np.pad(seq_idxs, (0,npad), mode="constant")
        batch_idxs = pad_idxs.reshape(r_lead, -1)  # (n // (b*2), b*2)
        np.random.shuffle(batch_idxs)
        batch_idxs = batch_idxs.flatten().tolist()[:-npad]
        yield from batch_idxs


class DataLoader_GTrans(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        **kwargs
    ):
        """ Set shuffling to false by default, as shuffling is done manually at the start. """
        if shuffle:
            shuffle = False
            sampler = LengthGrouper(
                dataset, group_len=kwargs.get("group_len", True), batch_size=batch_size, delta=kwargs.get("delta", 10)
            )
        super(DataLoader_GTrans, self).__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            **kwargs
        )
        self.featurizer = collate_fn
