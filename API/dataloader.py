import copy
import os
from functools import partial

from .cath_dataset import CATH
from .dataloader_gtrans import DataLoader_GTrans
from .featurizer import featurize_GTrans
from .ts_dataset import TS


def load_data(
    data_name: str,
    batch_size: int,
    data_root: str,
    num_workers: int = os.cpu_count(),
    **kwargs
):
    if data_name == "CATH" or data_name == "TS":
        # load once and use shallow copies to reduce transfers and RAM
        cath_set = CATH(os.path.join(data_root, "cath"), mode="train", test_name="All")
        train_set, valid_set, test_set = map(lambda x: copy.copy(x), [cath_set] * 3)
        valid_set.change_mode("valid")
        test_set.change_mode("test")
        if data_name == "TS":
            test_set = TS(osp.join(data_root, "ts"))
        collate_fn = featurize_GTrans

    default_loader = partial(
        DataLoader_GTrans,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    trian_loader = default_loader(train_set, shuffle=True)
    valid_loader = default_loader(valid_set, shuffle=False)
    test_loader = default_loader(test_set, shuffle=False)

    return train_loader, valid_loader, test_loader


def make_cath_loader(test_set, batch_size, max_nodes=3000, num_workers=8):
    collate_fn = featurize_GTrans
    test_loader = DataLoader_GTrans(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return test_loader
