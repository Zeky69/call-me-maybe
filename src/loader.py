import json 
from .logger import logger
from .models import FunctionDef

def load_functions(file_path) -> list[FunctionDef]:

    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            functions = [FunctionDef(**item) for item in data]
            return functions
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while loading functions: {str(e)}")
    finally:
        logger.info(f"{len(functions) if 'functions' in locals() else 0} functions loaded from {file_path}")


def load_prompt(file_path) -> list[str]:
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            prompts = [item["prompt"] for item in data]
            return prompts
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while loading prompts: {str(e)}")
    finally:
        logger.info(f"{len(prompts) if 'prompts' in locals() else 0} prompts loaded from {file_path}")
