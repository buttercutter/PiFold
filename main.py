from __future__ import annotations
import json
import logging
import os.path as osp
import pickle
import warnings

import torch
from typing import Any

warnings.filterwarnings("ignore")

import logging
from pathlib import Path

from API import Recorder
from methods import ProDesign
from utils import *

logger = logging.getLogger(__name__)


class Exp:
    def __init__(self, args, show_params=True):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        if show_params:
            print_log(output_namespace(self.args))


    def recursive_set_(self, arg_name: str, new_value: Any) -> Exp:
        """ Recursively sets arg to the module and nn.Module children """
        # to self
        if arg_name in self.args.__dict__:
            self.args.__dict__[arg_name] = new_value
        # to method
        if arg_name in self.method.args.__dict__:
            self.method.args.__dict__[arg_name] = new_value
        # to model
        if arg_name in self.method.model.args.__dict__:
            self.method.model.args.__dict__[arg_name] = new_value
        # to decoder. Attr, no longer a dict
        if arg_name in self.method.model.decoder.__dict__:
            self.method.model.decoder.__dict__[arg_name] = new_value
        # to encoder
        if arg_name in self.method.model.encoder.__dict__:
            self.method.model.encoder.__dict__[arg_name] = new_value
        # to encoder layers
        for i,layer in enumerate(self.method.model.encoder.encoder_layers):
            if arg_name in layer.__dict__:
                self.method.model.encoder.encoder_layers[i].__dict__[arg_name] = new_value
            # to encoder edge_update
            if arg_name in layer.edge_update.__dict__:
                self.method.model.encoder.encoder_layers[i].edge_update.__dict__[arg_name] = new_value
                self.method.model.encoder.encoder_layers[i].edge_update.norm.__dict__[arg_name] = new_value
            # to encoder context
            if arg_name in layer.context.__dict__:
                self.method.model.encoder.encoder_layers[i].context.__dict__[arg_name] = new_value
            # to encoder context
            if arg_name in layer.attention.__dict__:
                self.method.model.encoder.encoder_layers[i].attention.__dict__[arg_name] = new_value
            # to encoder norms
            for j, norm in enumerate(layer.norm):
                if arg_name in norm.__dict__:
                    self.method.model.encoder.encoder_layers[i].norm[j].__dict__[arg_name] = new_value
        # to embedding norms
        if arg_name in self.method.model.norm_nodes.__dict__:
            self.method.model.norm_nodes.__dict__[arg_name] = new_value
        if arg_name in self.method.model.norm_edges.__dict__:
            self.method.model.norm_edges.__dict__[arg_name] = new_value
        if arg_name in self.method.model.W_v_norm1.__dict__:
            self.method.model.W_v_norm1.__dict__[arg_name] = new_value
        if arg_name in self.method.model.W_v_norm2.__dict__:
            self.method.model.W_v_norm2.__dict__[arg_name] = new_value

        return self

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device("cuda:0")
            print("Use GPU:", device)
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, "checkpoints")
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, "model_param.json")
        with open(sv_param, "w") as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            filename=osp.join(self.path, "log.log"),
            filemode="a",
            format="%(asctime)s - %(message)s",
        )

        self._get_data()
        self._build_method()

    def _build_method(self):
        steps_per_epoch = 1000 # 2000  # 1000
        if self.args.method == "ProDesign":
            self.method = ProDesign(self.args, self.device, steps_per_epoch)

        # load a pretrained model
        if self.args.from_pretrained is not None:
            if Path.isfile(self.args.pretrained_model):
                logger.info(f"Loading checkpoint '{self.args.pretrained_model}'")
                self.method.model.load_state_dict(
                    torch.load(self.args.pretrained_model, map_location=self.device)
                )
            else:
                logger.error(
                    f"Could not load '{self.args.pretrained_model}' and starting from scratch"
                )

    def _get_data(self):
        data = get_dataset(self.config)
        self.train_loader, self.valid_loader, self.test_loader = data

    def train(self):
        recorder = Recorder(self.args.patience, verbose=True)
        for epoch in range(self.args.epoch):
            train_loss, train_perplexity = self.method.train_one_epoch(
                self.train_loader
            )

            if self.args.wandb_project:
                train_log = {
                    "Train/Loss": train_loss,
                    "Train/Perplexity": train_perplexity,
                }
                valid_log = {}
                test_log = {}

            if epoch % self.args.log_step == 0:
                with torch.no_grad():
                    valid_loss, valid_perplexity = self.valid()
                    # self._save(name=str(epoch))

                print_log(
                    (
                        f"Epoch: {epoch + 1}, Steps: {len(self.train_loader)} | ",
                        f"Train Loss: {train_loss:.4f}, Train Perp: {train_perplexity:.4f} | ",
                        f"Valid Loss: {valid_loss:.4f}, Valid Perp: {valid_perplexity:.4f}",
                    )
                )

                recorder(valid_loss, self.method.model, self.path)

                if self.args.test_every_epoch or recorder.counter == 0:
                    test_perplexity, test_recovery, test_subcat_recovery = self.test()
                else:
                    test_subcat_recovery = {}
                    test_perplexity, test_recovery = np.nan, np.nan

                if self.args.wandb_project:
                    valid_log = {
                        "Val/Loss": valid_loss,
                        "Val/Perplexity": valid_perplexity,
                    }
                    test_log = {
                        "Test/Perplexity": test_perplexity,
                        "Test/Recovery": test_recovery,
                        **{
                            f"Test/{cat}/Recovery": val
                            for cat, val in test_subcat_recovery.items()
                        },
                    }

            if self.args.wandb_project:
                wandb.log(
                    {
                        **train_log,
                        **valid_log,
                        **test_log,
                        "lr": self.method.scheduler.get_lr()[0],
                    }
                )

            if epoch % self.args.log_step and recorder.early_stop:
                print("Early stopping")
                logging.info("Early stopping")
                break

        best_model_path = osp.join(self.path, "checkpoint.pth")
        self.method.model.load_state_dict(torch.load(best_model_path))

    def valid(self):
        valid_loss, valid_perplexity = self.method.valid_one_epoch(self.valid_loader)

        print_log("Valid Perp: {0:.4f}".format(valid_perplexity))

        return valid_loss, valid_perplexity

    def test(self):
        (
            test_perplexity,
            test_recovery,
            test_subcat_recovery,
        ) = self.method.test_one_epoch(self.test_loader)
        print_log(
            "Test Perp: {0:.4f}, Test Rec: {1:.4f}\n".format(
                test_perplexity, test_recovery
            )
        )

        for cat, val in test_subcat_recovery.items():
            print_log("Category {0} Rec: {1:.4f}\n".format(cat, val))

        return test_perplexity, test_recovery, test_subcat_recovery

    def init_logger(self, config: dict) -> None:
        if self.args.wandb_project:
            wandb.init(project=self.args.wandb_project, config=config)

    def end_logger(self, test_perp: float, test_rec: float) -> None:
        if self.args.wandb_project:
            wandb.summary["test_perplexity"] = test_perp
            wandb.summary["test_recovery"] = test_rec
            wandb.finish()


if __name__ == "__main__":
    from parser import create_parser

    args = create_parser().parse_args()
    config = args.__dict__

    if args.wandb_project:
        try:
            import wandb
        except ImportError as e:
            logger.error(f"Could not import wandb: {e}. Consider `pip install wandb`")

    print(config)

    exp = Exp(args)
    exp.init_logger(config)

    # svpath = '/gaozhangyang/experiments/ProDesign/results/ProDesign/'
    # exp.method.model.load_state_dict(torch.load(svpath+'checkpoint.pth'))

    print(">>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    exp.train()

    print(">>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    test_perp, test_rec, test_subcat_recovery = exp.test()
    exp.end_logger(test_perp, test_rec)
