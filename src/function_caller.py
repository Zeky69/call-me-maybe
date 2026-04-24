from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import Any
from .vocabulary import Vocabulary
from .constrained_decoder import ConstrainedDecoder
from .models import FunctionDef, FunctionCall


class FunctionCaller:
    def __init__(
            self,
            model: Small_LLM_Model,
            functions: list[FunctionDef]) -> None:
        self.model: Small_LLM_Model = model
        self.functions: list[FunctionDef] = functions
        self.vocab: Vocabulary = Vocabulary(model.get_path_to_vocab_file())
        self.decoder: ConstrainedDecoder = ConstrainedDecoder(
            model, self.vocab)

    def _build_prompt_system(self) -> str:
        prompt = (
            "<|im_start|>system\n"
            "You are a function calling assistant. "
            "Choose the right function.\n"
            "Here are the available functions:\n"
        )
        for fn in self.functions:
            param_list = ", ".join(
                f"{name}: {p.type}"
                for name, p in fn.parameters.items()
            )
            prompt += f"- {fn.name}({param_list})\n"
        prompt += "<|im_end|>\n"
        return prompt

    def _build_prompt_function(
            self,
            prompt_user: str,
            function_def: FunctionDef) -> str:
        prompt = (
            "<|im_start|>system\n"
            "You are a function calling assistant. Extract the INPUT "
            "parameters from the user message to call the function.\n"
            "IMPORTANT: Do NOT compute the result. Extract the raw "
            "inputs that should be PASSED to the function.\n"
            f"Function: {function_def.name} — {function_def.description}\n"
            "Parameters:\n"
        )
        for name, p in function_def.parameters.items():
            prompt += f"- {name} ({p.type})\n"
        prompt += (
            "Respond with a JSON object containing the parameter values.\n"
            "IMPORTANT: If the user message contains double quotes \"\", "
            "keep them in the extracted string value using \\\".\n"
        )
        if any(True for p in function_def.parameters.keys()
                if "regex" in p.lower()):
            prompt += "IMPORTANT: For replacement " \
                "strings, always use the literal "
            "symbol, never its English name: "
            "asterisks → \"*\", hash → \"#\", dash → \"-\".\n"
        prompt += "<|im_end|>\n"

        prompt += f"<|im_start|>user\n{prompt_user}\n<|im_end|>\n"
        prompt += "<|im_start|>assistant\n<think></think>\n"

        return prompt

    def process(self, prompt: str, pbar: Any = None) -> FunctionCall:
        def _status(msg: str) -> None:
            if pbar is not None:
                pbar.set_postfix_str(msg)

        system_prompt = self._build_prompt_system()
        full_prompt = system_prompt + \
            f"<|im_start|>user\n{prompt}\n<|im_end|>" + \
            "\n<|im_start|>assistant\n<think></think>\n"

        _status("selecting function...")
        input_ids = self.model.encode(full_prompt)[0].tolist()

        candidates = [f.name for f in self.functions]
        function_name = self.decoder.generate_one_of(input_ids, candidates)
        _status(f"function: {function_name}")

        function_def = next(
            f for f in self.functions if f.name == function_name)
        if function_def is None:
            raise ValueError(f"Function {function_name} not found")

        params: dict[str, float | int | bool | str] = {}
        prompt_function = self._build_prompt_function(prompt, function_def)
        prompt_function += '{'
        param_items = list(function_def.parameters.items())
        for i, (param_name, param_def) in enumerate(param_items):
            _status(
                f"{function_name} > filling '{param_name}' ("
                f"{param_def.type})")
            prompt_function += '# parameter \n'
            prompt_function += f'"{param_name}": '
            if param_def.type == "string":
                prompt_function += '"'
            input_ids = self.model.encode(prompt_function)[0].tolist()
            param_value: float | int | bool | str
            if param_def.type == "number":
                param_value = self.decoder.generate_number(input_ids)
                prompt_function += str(param_value)
            elif param_def.type == "integer":
                param_value = int(self.decoder.generate_number(input_ids))
                prompt_function += str(param_value)
            elif param_def.type == "boolean":
                param_value = self.decoder.generate_bool(input_ids)
                prompt_function += str(param_value).lower()
            elif param_def.type == "string":
                param_value = self.decoder.generate_string(
                    input_ids, stop_char='"')
                prompt_function += f'{param_value}"'
            else:
                raise ValueError("Unsupported parameter type"
                                 f": {param_def.type}")
            params[param_name] = param_value
            if i < len(param_items) - 1:
                prompt_function += ', '
        prompt_function += '}'
        return FunctionCall(
            prompt=prompt,
            name=function_name,
            parameters=params)
