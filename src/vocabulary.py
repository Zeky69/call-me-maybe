import json
from typing import Any
from .logger import logger


class Vocabulary:
    def __init__(self, path: str) -> None:
        self._vocab_by_string: dict[str, int] = {}
        self._vocab_by_id: dict[int, str] = {}
        self._path = path
        self.load()

    def _clean_token(self, token: str) -> str:
        token = token.replace("Ġ", " ")
        token = token.replace("Ċ", "\n")
        return token

    def load(self) -> None:
        try:
            with open(self._path, "r") as file:
                data: Any = json.load(file)

            self._vocab_by_id = {
                int(v): self._clean_token(str(k))
                for k, v in data.items()
            }
            self._vocab_by_string = {
                str(k): int(v)
                for k, v in data.items()
            }
        except FileNotFoundError:
            logger.error(f"File not found: {self._path}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file: {self._path}")
        except Exception as e:
            logger.error(
                f"An error occurred while loading vocabulary: "
                f"{str(e)}"
            )
        finally:
            logger.info(
                f"{len(self._vocab_by_string)} vocabulary entries "
                f"loaded from {self._path}"
            )

    def get_id(self, string: str) -> int:
        return self._vocab_by_string.get(string, -1)

    def get_string(self, id: int) -> str:
        return self._vocab_by_id.get(id, "")

    def get_tokens(self) -> dict[int, str]:
        return self._vocab_by_id

    def find_tokens_by_prefix(self, prefix: str) -> list[int]:
        return [id for id, s in self._vocab_by_id.items()
                if s.startswith(prefix)]
