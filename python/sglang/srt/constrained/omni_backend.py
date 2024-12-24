"""Constrained decoding with Omni backend."""

import re
import logging
from typing import List, Tuple
from functools import lru_cache

import torch
import numpy as np

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
)

logger = logging.getLogger(__name__)


class OmniGrammar(BaseGrammarObject):

    def __init__(
        self, mask: torch.Tensor
    ) -> None:
        self.mask = mask

    def accept_token(self, token: int):
        pass

    def try_jump_forward(self, tokenizer) -> Tuple[List[int], str]:
        return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        pass

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        pass

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        vocab_mask[idx] = self.mask.to(vocab_mask.device, non_blocking=True)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        logits.masked_fill_(vocab_mask, float("-inf"))

    def copy(self):
        return OmniGrammar(self.mask)


class OmniGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.decoded_vocab = self._get_decoded_vocab()
        self.finished = False

    def _get_decoded_vocab(self) -> List[str]:
        """
        Decode the entire vocabulary at once.
        """
        indices = torch.arange(self.vocab_size)
        return self.tokenizer.batch_decode(indices.unsqueeze(-1))
    
    def create_mask(
        self,
        pattern: str
    ) -> torch.Tensor:
        """
        NumPy-based implementation for creating vocabulary mask.
        
        Args:
            pattern: Regex pattern to match against decoded tokens
        
        Returns:
            torch.Tensor mask of shape (vocab_size,)
        """
        regex = re.compile(pattern)
        
        # Create mask
        mask = np.array([bool(regex.search(token)) for token in self.decoded_vocab])
        mask_tensor = torch.from_numpy(mask)
            
        return mask_tensor

    def init_value_impl(self, key: Tuple[str, str]) -> OmniGrammar:
        key_type, key_string = key
        if key_type == "regex":
            try:
                mask = self.create_mask(key_string)
            except RuntimeError as e:
                logging.warning(
                    f"Skip invalid regex pattern: {key_string}, {e=}"
                )
                return None
        elif key_type == "json":
            logger.warning(
                "regex hasn't been supported by xgrammar yet. This is skipped."
            )
            return None
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

        return OmniGrammar(mask)

