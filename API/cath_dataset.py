from __future__ import annotations

import copy
import json
import os

import numpy as np
import torch.utils.data as data
from tqdm import tqdm

from .featurizer import ALPHABET
from .utils import cached_property


class CATH(data.Dataset):
    def __init__(
        self, path="./", mode="train", max_length=500, test_name="All", data=None
    ):
        self.path = path
        self.mode = mode
        self.max_length = max_length
        self.test_name = test_name
        if data is None:
            self.data = self.cache_data[mode]
        else:
            self.data = data

    @cached_property
    def cache_data(self):
        alphabet_set = set(ALPHABET)
        if not os.path.exists(self.path):
            raise "no such file:{} !!!".format(self.path)
        else:
            with open(self.path + "/chain_set.jsonl") as f:
                lines = f.readlines()
            data_list = []
            for line in tqdm(lines):
                entry = json.loads(line)
                seq = entry["seq"]

                bad_chars = set(seq).difference(alphabet_set)

                # only keep prots w/ standard residues
                # TODO: hypnopump@ think about masking out at loss compute and use these datapoints
                if len(bad_chars) == 0:
                    if len(entry["seq"]) <= self.max_length:
                        data_list.append(
                            {
                                "title": entry["name"],
                                "seq": entry["seq"],
                                **{
                                    k: np.asarray(v) for k, v in entry["coords"].items()
                                },
                            }
                        )

            with open(self.path + "/chain_set_splits.json") as f:
                dataset_splits = json.load(f)

            if self.test_name == "L100":
                with open(self.path + "/test_split_L100.json") as f:
                    test_splits = json.load(f)
                dataset_splits["test"] = test_splits["test"]

            if self.test_name == "sc":
                with open(self.path + "/test_split_sc.json") as f:
                    test_splits = json.load(f)
                dataset_splits["test"] = test_splits["test"]

            name2set = {}
            name2set.update({name: "train" for name in dataset_splits["train"]})
            name2set.update({name: "valid" for name in dataset_splits["validation"]})
            name2set.update({name: "test" for name in dataset_splits["test"]})

            data_dict = {"train": [], "valid": [], "test": []}
            for data in data_list:
                if name2set.get(data["title"]):
                    if name2set[data["title"]] == "train":
                        data_dict["train"].append(data)

                    if name2set[data["title"]] == "valid":
                        data_dict["valid"].append(data)

                    if name2set[data["title"]] == "test":
                        data["category"] = "Unkown"
                        data["score"] = 100.0
                        data_dict["test"].append(data)
            return data_dict

    def change_mode(self, mode):
        self.data = self.cache_data[mode]

    def __len__(self):
        return len(self.data)

    def get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]

    def copy(self, deep: bool = False) -> CATH:
        return copy.copy(self) if not deep else copy.deepcopy(self)
