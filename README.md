*This project has been created as part of the 42 curriculum by zakburak.*

# Call Me Maybe — Introduction to Function Calling in LLMs

## Description

**Call Me Maybe** is a function-calling system that translates natural language prompts into structured, machine-executable function calls using a small Large Language Model (LLM).

Given a prompt like `"What is the product of 3 and 5?"`, the system does **not** return the answer `15`. Instead, it produces:

```json
{
  "prompt": "What is the product of 3 and 5?",
  "name": "fn_multiply_numbers",
  "parameters": {"a": 3.0, "b": 5.0}
}
```

The key challenge is reliability: small language models (≤ 1B parameters) only produce valid JSON ~30% of the time when prompted naively. This project achieves **100% valid JSON output** through **constrained decoding** — a technique that guides the model token-by-token, masking any tokens that would violate the output schema before each selection step.

The default model is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (600M parameters).

---

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
make install
# or equivalently:
uv sync
```

This installs all dependencies, including `numpy`, `pydantic`, `tqdm`, and `llm_sdk` (the local LLM wrapper).

### Running

```bash
# Default paths (data/input/ → data/output/)
make run

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Debug mode (pdb)

```bash
make debug
```

### Lint

```bash
make lint         # flake8 + mypy with standard flags
make lint-strict  # flake8 + mypy --strict
```

### Clean

```bash
make clean   # removes __pycache__, .mypy_cache, *.pyc
make fclean  # also removes .venv and uv.lock
```

---

## Algorithm Explanation — Constrained Decoding

### Overview

Standard LLM decoding selects the next token from a probability distribution over the entire vocabulary (~150k tokens for Qwen). For structured output, most of those tokens are invalid at any given position.

Constrained decoding **modifies the logit vector before token selection**:

1. The model produces raw logits over the full vocabulary.
2. The system determines which tokens are **valid** at the current generation position (based on JSON structure + schema constraints).
3. Logits for all **invalid tokens** are set to `-inf`.
4. The highest-scoring remaining token is selected (argmax).
5. This token is appended to the sequence and the process repeats.

### Generation Steps

For each prompt, the pipeline is:

```
1. Build system prompt listing available functions
2. [Function selection]  generate_one_of(candidates=function_names)
3. Look up the chosen function's parameter schema
4. Build a focused prompt for parameter extraction
5. For each parameter, call the matching generator:
   - number   → generate_number()   (allows digits, '.', '-')
   - integer  → generate_integer()  (same, then cast to int)
   - boolean  → generate_bool()     (restricted to "true"/"false")
   - string   → generate_string()   (reads until closing '"')
6. Assemble FunctionCall object and write to output JSON
```

### generate_one_of

Performs prefix-matching constrained decoding. At each step, only tokens that extend a prefix of at least one remaining candidate are allowed. Generation stops when exactly one candidate matches the accumulated output.

### generate_number / generate_integer

Uses a pre-compiled regex (`^[0-9.\-]+$`) to filter the vocabulary to numeric-only tokens at startup. At each step, the best unconstrained token is inspected first: if it is non-numeric, generation stops (the model is naturally done). Otherwise, only numeric tokens compete.

### generate_bool

Delegates to `generate_one_of(["true", "false"])` — a binary constrained choice.

### generate_string

Iterates token-by-token, masking tokens that contain `"` or newline characters. Stops as soon as a stop character is detected. The accumulated string is passed through `json.loads('"…"')` to handle JSON escape sequences correctly.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **argmax instead of sampling** | Deterministic, reproducible results. Sampling adds randomness without benefit when constrained. |
| **Pre-compute numeric token IDs** | Building the numeric token set once at `ConstrainedDecoder` init avoids repeated regex scans over 150k tokens per generation step. |
| **NumPy for logit masking** | Vectorised `-inf` assignment over a flat float32 array is ~10× faster than a Python loop. |
| **Focused parameter prompts** | A dedicated prompt for each parameter ("Extract the `name` (string) parameter from the user message") outperforms a single combined prompt on small models. |
| **Pydantic for all data models** | Ensures schema validation at parse time (`FunctionDef`, `ParameterDef`, `FunctionCall`), making errors immediately visible rather than surfacing later. |
| **`logger.error` exits immediately** | Errors in file loading or model I/O are unrecoverable; exiting with a clear message is safer than propagating a half-initialised state. |
| **LOG_LEVEL env var** | Matches the Makefile convention (`LOG_LEVEL ?= ERROR`). Pass `LOG_LEVEL=DEBUG make run` for verbose output without touching source code. |

---

## Performance Analysis

Tested on the private test set (11 prompts, 6 functions):

| Metric | Result |
|---|---|
| JSON validity | **100%** — every output is parseable |
| Function selection accuracy | **~100%** — constrained decoding forces a valid function name |
| Argument extraction accuracy | **≥ 90%** — depends on prompt clarity |
| Processing time | ~15–30 s per prompt on CPU (Qwen3-0.6B) |
| Total runtime for 11 prompts | < 5 min on standard hardware |

The constrained decoder guarantees structural correctness. Semantic correctness (choosing the right function, extracting the right value) depends on the LLM's understanding of the prompt — the 0.6B model handles straightforward cases reliably.

---

## Challenges Faced

**1. Implicit `None` returns from error handlers**
`loader.py` originally used a `finally` block that referenced variables potentially not yet defined. Refactored to initialise variables before `try` and log inside the `try` block after assignment.

**2. Hardcoded log level**
`__main__.py` initially set `logger.set_level("DEBUG")` unconditionally, polluting all output. Fixed by reading `LOG_LEVEL` from the environment, consistent with the Makefile.

**3. Token boundary alignment**
Tokens don't always align with word boundaries. A function name like `fn_multiply_numbers` may be split into `fn`, `_mult`, `iply`, `_numbers`. `generate_one_of` handles this by checking whether `output + token` is still a prefix of any candidate, rather than matching whole words.

**4. Number generation stop condition**
A pure constrained approach for numbers (only allow numeric tokens at every step) caused the model to loop indefinitely on some inputs. The solution: check the model's **unconstrained** best token first — if it is non-numeric, the model is signalling completion.

**5. String escaping**
JSON strings can contain escape sequences (`\n`, `\"`, etc.). Simply concatenating raw tokens and splitting on `"` was unreliable. Using `json.loads('"' + accumulated + '"')` correctly handles all escape sequences.

---

## Testing Strategy

1. **End-to-end run** on the provided public test set (`data/input/`) and verify the output JSON is valid and matches expected function calls.
2. **Manual edge cases**: empty strings, negative numbers, large floats, boolean arguments, multi-parameter functions.
3. **Type checking**: `make lint` (flake8 + mypy) catches type annotation issues before runtime.
4. **Schema validation**: Pydantic raises `ValidationError` if any output fails the `FunctionCall` schema — caught and reported during the tqdm loop.

---

## Example Usage

```bash
# Run with default files
uv run python -m src

# Run with custom files
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

# Run with verbose logging
LOG_LEVEL=INFO make run

# Run with debug logging
LOG_LEVEL=DEBUG make run
```

### Example input (`function_calling_tests.json`)

```json
[
  {"prompt": "What is the product of 3 and 5?"},
  {"prompt": "Is 42 an even number?"},
  {"prompt": "Read the file /etc/hosts with utf-8 encoding"}
]
```

### Example output (`function_calling_results.json`)

```json
[
  {
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "parameters": {"a": 3.0, "b": 5.0}
  },
  {
    "prompt": "Is 42 an even number?",
    "name": "fn_is_even",
    "parameters": {"n": 42.0}
  },
  {
    "prompt": "Read the file /etc/hosts with utf-8 encoding",
    "name": "fn_read_file",
    "parameters": {"path": "/etc/hosts", "encoding": "utf-8"}
  }
]
```

---

## Resources

### References

- [Qwen3 Model Card — Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained Decoding / Structured Generation — Outlines library concepts](https://github.com/outlines-dev/outlines)
- [JSON Schema specification](https://json-schema.org/)
- [Byte-Pair Encoding tokenisation (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909)
- [Function Calling in OpenAI API — official docs](https://platform.openai.com/docs/guides/function-calling)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)

### AI Usage

GitHub Copilot (Claude Sonnet) was used during this project for:

- **Reviewing existing code** for type annotation completeness and flake8 compliance.
- **Identifying silent error-handling bugs** in `loader.py` (undefined variable in `finally` block).
- **Writing the README** — structure, section content, and example formatting.
- **Explaining constrained decoding concepts** to clarify the algorithm description.

All AI-generated content was reviewed, understood, and tested before inclusion.
