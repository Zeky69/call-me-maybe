from .logger import logger
import argparse
from tqdm import tqdm
from .loader import load_functions, load_prompt
from .vocabulary import Vocabulary
from .constrained_decoder import ConstrainedDecoder
from .models import FunctionDef, FunctionCall
from llm_sdk import Small_LLM_Model
from .function_caller import FunctionCaller
from .writer import save_results

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def main():
    parser = argparse.ArgumentParser(description="Function calling assistant")
    parser.add_argument("--functions_definition", type=str, default=DEFAULT_FUNCTIONS, help="Path to the functions definition file")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to the input file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to the output file")
    args = parser.parse_args()

    functions = load_functions(args.functions_definition)
    prompts = load_prompt(args.input)

    model = Small_LLM_Model()
    function_caller = FunctionCaller(model, functions)
    results = []
    with tqdm(prompts, desc="Processing prompts") as pbar:
        for prompt in pbar:
            try:
                result = function_caller.process(prompt, pbar)
                results.append(result)
            except Exception as e:
                logger.error(f"An error occurred while processing prompt '{prompt}': {str(e)}")
            finally:
                pbar.set_postfix_str("")
    save_results(args.output, results)
    logger.info(f"Processing completed. Results saved to {args.output}")

if __name__ == "__main__":
    logger.set_level("DEBUG")
    main()