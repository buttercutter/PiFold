from __future__ import annotations

import copy
import json
import os

import numpy as np
import torch.utils.data as data


class TS(data.Dataset):
    def __init__(self, path="./"):
        if not os.path.exists(path):
            raise ValueError(f"no such file:{path} !!!")
        else:
            ts50_data = json.load(open(path + "/ts50.json"))
            ts500_data = json.load(open(path + "/ts500.json"))

        # TS500 has proteins with lengths of 500+
        # TS50 only contains proteins with lengths less than 500
        self.data = []
        for temp in ts50_data:
            coords = np.array(temp["coords"])
            self.data.append(
                {
                    "title": temp["name"],
                    "seq": temp["seq"],
                    "CA": coords[:, 1, :],
                    "C": coords[:, 2, :],
                    "O": coords[:, 3, :],
                    "N": coords[:, 0, :],
                    "category": "ts50",
                }
            )

        for temp in ts500_data:
            coords = np.array(temp["coords"])
            self.data.append(
                {
                    "title": temp["name"],
                    "seq": temp["seq"],
                    "CA": coords[:, 1, :],
                    "C": coords[:, 2, :],
                    "O": coords[:, 3, :],
                    "N": coords[:, 0, :],
                    "category": "ts500",
                }
            )

        self.data = sorted(self.data, key=lambda x: len(x["seq"]))

    def __len__(self):
        return len(self.data)

    def get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]

    def copy(self, deep: bool = False) -> TS:
        return copy.copy(self) if not deep else copy.deepcopy(self)
