import logging
from typing import Union

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, is_flashinfer_available

if is_flashinfer_available():
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


logger = logging.getLogger(__name__)


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detectioin = global_server_args_dict["enable_nan_detection"]

    def forward(
        self,
        logits: Union[torch.Tensor, LogitsProcessorOutput],
        sampling_info: SamplingBatchInfo,
    ):
        if isinstance(logits, LogitsProcessorOutput):
            logits = logits.next_token_logits

        logits = logits.contiguous()

        if self.use_nan_detectioin and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = torch.argmax(logits, -1)
        else:
            # Post process logits
            logits.div_(sampling_info.temperatures)

            if any(sampling_info.top_n_sigmas > 0):
                logits = apply_top_n_sigma_to_logits_torch(
                    logits, sampling_info.top_n_sigmas
                )

            probs = torch.softmax(logits, dim=-1)
            del logits

            if global_server_args_dict["sampling_backend"] == "flashinfer":
                max_top_k_round, batch_size = 32, probs.shape[0]
                uniform_samples = torch.rand(
                    (max_top_k_round, batch_size), device=probs.device
                )
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids, success = min_p_sampling_from_probs(
                        probs, uniform_samples, sampling_info.min_ps
                    )
                else:
                    batch_next_token_ids, success = top_k_top_p_sampling_from_probs(
                        probs,
                        uniform_samples,
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                    )

                if self.use_nan_detectioin and not torch.all(success):
                    logger.warning("Detected errors during sampling!")
                    batch_next_token_ids = torch.zeros_like(batch_next_token_ids)
            elif global_server_args_dict["sampling_backend"] == "pytorch":
                # A slower fallback implementation with torch native operations.
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                )
            else:
                raise ValueError(
                    f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
                )

        return batch_next_token_ids.to(torch.int32)
    
def apply_top_n_sigma_to_logits_torch(logits: torch.Tensor, top_n_sigmas: torch.Tensor):
    max_logit = torch.max(logits, dim=-1, keepdim=True).values
    sigma = torch.std(logits, dim=-1, keepdim=True)
    # Create mask and enable only for the requests that have top_n_sigma > 0
    mask = (top_n_sigmas.view(-1, 1) <= 0) | (
        logits >= max_logit - top_n_sigmas.view(-1, 1) * sigma
    )

    # Apply mask
    logits = torch.where(mask, logits, torch.tensor(float("-inf")).to(logits.device))
    return logits


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_p_normalize_probs(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    if global_server_args_dict["sampling_backend"] == "flashinfer":
        return top_p_renorm_prob(probs, top_ps)
    elif global_server_args_dict["sampling_backend"] == "pytorch":
        # See also top_k_top_p_min_p_sampling_from_probs_torch
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)
    else:
        raise ValueError(
            f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
        )
