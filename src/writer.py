import json
import os
from .logger import logger
from .models import FunctionCall

def save_results(file_path:str, results:list[FunctionCall]):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        data = [result.model_dump() for result in results]
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        logger.info(f"Results saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving results: {str(e)}")