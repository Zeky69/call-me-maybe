VENV	= .venv
MAIN	= src
ARGS	=
LOG_LEVEL ?= ERROR


.PHONY: all install run debug lint lint-strict  clean fclean re

all: run

$(VENV): pyproject.toml
	uv sync

install: $(VENV)

run: $(VENV)
	LOG_LEVEL=$(LOG_LEVEL) uv run -m $(MAIN) $(ARGS)

debug: $(VENV)
	LOG_LEVEL=$(LOG_LEVEL) uv run python -m pdb -m $(MAIN) $(ARGS)

lint: $(VENV)
	uv run flake8 .
	uv run mypy src/ \
		--warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: $(VENV)
	uv run flake8 . --exclude=.venv,build,dist,llm_sdk
	uv run mypy src/ --strict


clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

fclean: clean
	rm -rf $(VENV)
	rm -f uv.lock

re: fclean all
