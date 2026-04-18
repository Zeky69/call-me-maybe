import json
from .logger import logger
from .models import FunctionDef


def load_functions(file_path: str) -> list[FunctionDef]:

    functions: list[FunctionDef] = []
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            functions = [FunctionDef(**item) for item in data]
        logger.info(f"{len(functions)} functions loaded from {file_path}")
        return functions
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while loading functions: {str(e)}")


def load_prompt(file_path: str) -> list[str]:
    prompts: list[str] = []
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            prompts = [item["prompt"] for item in data]
        logger.info(f"{len(prompts)} prompts loaded from {file_path}")
        return prompts
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while loading prompts: {str(e)}")
