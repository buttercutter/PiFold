from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from tqdm import tqdm

from .base_method import Base_method
from .prodesign_model import ProDesign_Model
from .utils import cuda


class ProDesign(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.transfer_func = lambda x: x
        if self.args.use_gpu:
            self.transfer_func = partial(cuda, device=self.device)

    def _build_model(self):
        return ProDesign_Model(self.args).to(self.device)

    def train_one_epoch(self, train_loader):
        self.model.train()
        train_sum, train_weights = 0, 1e-7

        train_pbar = tqdm(train_loader)
        for step_idx, batch in enumerate(train_pbar):
            self.optimizer.zero_grad()
            X, S, score, mask, lengths = self.transfer_func(batch)
            (
                X,
                S,
                score,
                h_V,
                h_E,
                E_idx,
                batch_id,
                mask_bw,
                mask_fw,
                decoding_order,
                node_mask,
                edge_mask,
            ) = self.model._get_features(
                S, score, X=X, mask=mask, mode=self.args.train_mode
            )
            log_probs = self.model(
                h_V,
                h_E,
                E_idx,
                batch_id,
                mode=self.args.train_mode,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
            if self.args.train_mode == "sparse":
                loss = self.criterion(log_probs, S).mean()
            elif self.args.train_mode == "dense":
                node_mask_unrolled = node_mask.reshape(-1)
                log_probs_unrolled = log_probs.reshape(-1, log_probs.shape[-1])
                S_unrolled = S.reshape(-1)
                loss = self.criterion(log_probs_unrolled, S_unrolled)
                loss = (loss * node_mask_unrolled).sum() / node_mask_unrolled.sum()
            else:
                raise NotImplementedError(
                    "Only sparse and dense modes are supported for now"
                )

            loss.backward()
            # TODO: hypnopump@ consider clipping gradients on a per-sample basis instead of per-batch. How ??? idk yet
            # TODO: hypnopump@ see UniFold paper for comparison, and OpenFold for an implementation
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            if isinstance(train_sum, int):
                train_sum = torch.sum(loss * mask)
                train_weights = torch.sum(mask)
            else:
                train_sum += torch.sum(loss * mask)
                train_weights += torch.sum(mask)

            if step_idx % self.args.display_step == 0:
                train_pbar.set_description("train loss: {:.4f}".format(loss.item()))

        train_loss = train_sum / train_weights
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.cpu().data.numpy()
        train_perplexity = np.exp(train_loss)
        return train_loss, train_perplexity

    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        valid_sum, valid_weights = 0, 1e-7

        valid_pbar = tqdm(valid_loader)
        with torch.no_grad():
            for step_idx, batch in enumerate(valid_pbar):
                X, S, score, mask, lengths = self.transfer_func(batch)
                (
                    X,
                    S,
                    score,
                    h_V,
                    h_E,
                    E_idx,
                    batch_id,
                    mask_bw,
                    mask_fw,
                    decoding_order,
                    node_mask,
                    edge_mask,
                ) = self.model._get_features(
                    S, score, X=X, mask=mask, mode=self.args.train_mode
                )
                log_probs = self.model(
                    h_V,
                    h_E,
                    E_idx,
                    batch_id,
                    mode=self.args.train_mode,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                )
                if self.args.train_mode == "sparse":
                    loss = self.criterion(log_probs, S).mean()
                elif self.args.train_mode == "dense":
                    node_mask_unrolled = node_mask.reshape(-1)
                    log_probs_unrolled = log_probs.reshape(-1, log_probs.shape[-1])
                    S_unrolled = S.reshape(-1)
                    loss = self.criterion(log_probs_unrolled, S_unrolled)
                    loss = (loss * node_mask_unrolled).sum() / node_mask_unrolled.sum()
                else:
                    raise NotImplementedError(
                        "Only sparse and dense modes are supported for now"
                    )

                # FIXME: hypnopump@ simplify
                if isinstance(valid_sum, int):
                    valid_sum = torch.sum(loss * mask)
                    valid_weights = torch.sum(mask)
                else:
                    valid_sum += torch.sum(loss * mask)
                    valid_weights += torch.sum(mask)

                if step_idx % self.args.display_step == 0:
                    valid_pbar.set_description(
                        "valid loss: {:.4f}".format(loss.mean().item())
                    )

            valid_loss = valid_sum / valid_weights
            if isinstance(valid_loss, torch.Tensor):
                valid_loss = valid_loss.cpu().data.numpy()
            valid_perplexity = np.exp(valid_loss)
        return valid_loss, valid_perplexity

    def test_one_epoch(self, test_loader):
        self.model.eval()
        test_sum, test_weights = 0, 1e-7
        test_pbar = tqdm(test_loader)

        with torch.no_grad():
            for step_idx, batch in enumerate(test_pbar):
                X, S, score, mask, lengths = self.transfer_func(batch)
                (
                    X,
                    S,
                    score,
                    h_V,
                    h_E,
                    E_idx,
                    batch_id,
                    mask_bw,
                    mask_fw,
                    decoding_order,
                    node_mask,
                    edge_mask,
                ) = self.model._get_features(
                    S, score, X=X, mask=mask, mode=self.args.train_mode
                )
                log_probs = self.model(
                    h_V,
                    h_E,
                    E_idx,
                    batch_id,
                    mode=self.args.train_mode,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                )
                loss_mask = node_mask if self.args.train_mode == "dense" else None
                loss, loss_av = self.loss_nll_flatten(S, log_probs, mask=loss_mask)
                # FIXME: hypnopump@ wtf ???
                mask = torch.ones_like(loss)

                if isinstance(test_sum, int):
                    test_sum = torch.sum(loss * mask)
                    test_weights = torch.sum(mask)
                else:
                    test_sum += torch.sum(loss * mask)
                    test_weights += torch.sum(mask)

                if step_idx % self.args.display_step == 0:
                    test_pbar.set_description(
                        "test loss: {:.4f}".format(loss.mean().item())
                    )

            test_recovery, test_subcat_recovery = self._cal_recovery(
                test_loader.dataset, test_loader.featurizer
            )

        test_loss = test_sum / test_weights
        if isinstance(test_loss, torch.Tensor):
            test_loss = test_loss.cpu().data.numpy()
        test_perplexity = np.exp(test_loss)
        return test_perplexity, test_recovery, test_subcat_recovery

    def _cal_recovery(self, dataset, featurizer):
        """This part runs the sparse encoding for now."""
        self.residue_type_cmp = torch.zeros(20, device="cuda:0")
        self.residue_type_num = torch.zeros(20, device="cuda:0")
        recovery = []
        subcat_recovery = {}
        with torch.no_grad():
            for protein in tqdm(dataset):
                p_category = (
                    protein["category"] if "category" in protein.keys() else "Unknown"
                )
                if p_category not in subcat_recovery.keys():
                    subcat_recovery[p_category] = []

                protein = featurizer([protein])
                X, S, score, mask, lengths = cuda(protein, device=self.device)
                (
                    X,
                    S,
                    score,
                    h_V,
                    h_E,
                    E_idx,
                    batch_id,
                    mask_bw,
                    mask_fw,
                    decoding_order,
                    node_mask,
                    edge_mask,
                ) = self.model._get_features(
                    S,
                    score,
                    X=X,
                    mask=mask,
                    mode="sparse",
                )
                log_probs = self.model(
                    h_V,
                    h_E,
                    E_idx,
                    batch_id,
                    mode="sparse",
                )
                S_pred = torch.argmax(log_probs, dim=1)
                cmp = S_pred == S
                recovery_ = cmp.float().mean().cpu().numpy()

                self.residue_type_cmp += scatter_sum(
                    cmp.float(), S.long(), dim=0, dim_size=20
                )
                self.residue_type_num += scatter_sum(
                    torch.ones_like(cmp.float()), S.long(), dim=0, dim_size=20
                )

                if np.isnan(recovery_):
                    recovery_ = 0.0

                subcat_recovery[p_category].append(recovery_)
                recovery.append(recovery_)

            for key in subcat_recovery.keys():
                subcat_recovery[key] = np.median(subcat_recovery[key])

        self.mean_recovery = np.mean(recovery)
        self.std_recovery = np.std(recovery)
        self.min_recovery = np.min(recovery)
        self.max_recovery = np.max(recovery)
        self.median_recovery = np.median(recovery)
        recovery = np.median(recovery)
        return recovery, subcat_recovery

    def loss_nll_flatten(self, S, log_probs, mask: Optional[torch.Tensor] = None):
        """Negative log probabilities"""
        criterion = torch.nn.NLLLoss(reduction="none")
        if mask is None:
            loss = criterion(log_probs, S)
        else:
            loss = criterion(log_probs[mask], S[mask])
        loss_av = loss.mean()
        return loss, loss_av

    def loss_nll_smoothed(
        self, S, log_probs, weight: float = 0.1, mask: Optional[torch.Tensor] = None
    ):
        """Negative log probabilities"""
        S_onehot = torch.nn.functional.one_hot(S, num_classes=20).float()
        S_onehot = S_onehot + weight / float(S_onehot.size(-1))
        S_onehot = S_onehot.mean(dim=-1)  # (b n c)
        loss = -(S_onehot * log_probs).sum(-1)
        if mask is not None:
            loss = loss[mask]
        loss_av = torch.sum(loss.mean())
        return loss, loss_av
