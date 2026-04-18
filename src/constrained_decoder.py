from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .vocabulary import Vocabulary
import numpy as np
import re
import json


class ConstrainedDecoder:

    NEG = float("-inf")
    _NUMBER_RE = re.compile(r'^[0-9.\-]+$')

    def __init__(self, model: Small_LLM_Model, vocab: Vocabulary):
        self.model: Small_LLM_Model = model
        self.vocab: Vocabulary = vocab
        self._vocab_items = list(
            vocab.get_tokens().items())  # [(id, str), ...]

        # Precompute numeric token ids as a sorted numpy array for fast
        # indexing
        self._number_token_ids = np.array(
            [
                t_id for t_id, t_str in self._vocab_items
                if t_str and self._NUMBER_RE.fullmatch(t_str)
            ],
            dtype=np.int64,
        )

    def generate_one_of(
            self,
            input_ids: list[int],
            candidates: list[str],
            max_tokens: int | None = None) -> str:
        output: str = ""
        currend_input_ids = list(input_ids)
        total_tokens = 0
        while True:
            remaining = [c for c in candidates if c.startswith(output)]
            if not remaining:
                break
            if len(remaining) == 1 and remaining[0] == output:
                break

            logits = self.model.get_logits_from_input_ids(currend_input_ids)
            logits_np = np.array(logits, dtype=np.float32)

            # Build constrained logits: start with all NEG, allow valid tokens
            logits_constrained = np.full(
                len(logits_np), self.NEG, dtype=np.float32)
            for t_id, t_str in self._vocab_items:
                if not t_str or t_id >= len(logits_np):
                    continue
                tmp = output + t_str
                if any(c.startswith(tmp) for c in remaining):
                    logits_constrained[t_id] = logits_np[t_id]

            if np.all(logits_constrained == self.NEG):
                break

            next_token_id = int(logits_constrained.argmax())
            next_token = self.vocab.get_string(next_token_id)
            output += next_token
            currend_input_ids.append(next_token_id)
            total_tokens += 1
            if max_tokens is not None and total_tokens >= max_tokens:
                break

        return output

    def generate_number(
            self,
            input_ids: list[int],
            max_tokens: int | None = None) -> float:
        output: str = ""
        currend_input_ids = list(input_ids)
        total_tokens = 0
        while True:
            logits = self.model.get_logits_from_input_ids(currend_input_ids)
            logits_np = np.array(logits, dtype=np.float32)

            # Stop if the model's unconstrained best token is not numeric
            best_token_id = int(logits_np.argmax())
            best_token_str = self.vocab.get_string(best_token_id)
            if not best_token_str or not self._NUMBER_RE.fullmatch(
                    best_token_str):
                break

            # Apply precomputed numeric token ids (safe regardless of logits
            # size)
            logits_constrained = np.full(
                len(logits_np), self.NEG, dtype=np.float32)
            valid_ids = self._number_token_ids[self._number_token_ids < len(
                logits_np)]
            logits_constrained[valid_ids] = logits_np[valid_ids]

            if np.all(logits_constrained == self.NEG):
                break

            next_token_id = int(logits_constrained.argmax())
            next_token = self.vocab.get_string(next_token_id)
            output += next_token
            currend_input_ids.append(next_token_id)
            total_tokens += 1
            if max_tokens is not None and total_tokens >= max_tokens:
                break
        try:
            return float(output)
        except ValueError:
            return float('nan')

    def generate_bool(
            self,
            input_ids: list[int],
            max_tokens: int | None = None) -> bool:
        output = self.generate_one_of(input_ids, ["True", "False"], max_tokens)
        return output == "True"

    def generate_integer(
            self,
            input_ids: list[int],
            max_tokens: int | None = None) -> int:
        output = self.generate_number(input_ids, max_tokens)
        return int(output)

    def generate_string(
            self,
            input_ids: list[int],
            max_tokens: int | None = None,
            stop_char: str = '"') -> str:
        output: str = ""
        currend_input_ids = list(input_ids)
        total_tokens = 0
        while True:
            logits = self.model.get_logits_from_input_ids(currend_input_ids)

            next_token_id = int(np.array(logits, dtype=np.float32).argmax())
            next_token = self.vocab.get_string(next_token_id)
            if next_token is None:
                break
            if stop_char in next_token:
                output += next_token.split(stop_char)[0]
                break
            if "\n" in next_token:
                output += next_token.split("\n")[0]
                break
            output += next_token

            currend_input_ids.append(next_token_id)
            total_tokens += 1
            if max_tokens is not None and total_tokens >= max_tokens:
                break
        try:
            return str(json.loads(f'"{output}"'))
        except (json.JSONDecodeError, ValueError):
            return output
