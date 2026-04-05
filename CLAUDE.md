# CLAUDE.md

## Project overview

Chiquito is a layer-by-layer LLM inference library for machines with limited VRAM. It loads HuggingFace transformer models one layer at a time onto the GPU, preloading weights into RAM for speed.

## Commands

```bash
uv run ruff check src/           # lint
uv run ruff format --check src/  # format check
uv run ruff check --fix src/     # auto-fix lint
uv run ruff format src/          # auto-fix format
uv run mypy                      # type checking
uv run pytest                    # tests
```

## Workflow for every change

After making changes to the code:

1. **Lint and format**: run `uv run ruff check --fix src/ && uv run ruff format src/` to fix issues, then verify with `uv run ruff check src/ && uv run ruff format --check src/`.
2. **Type check**: run `uv run mypy` and fix any errors.
3. **Tests**: write or update tests in `tests/` covering the changed code, then run `uv run pytest` and ensure everything passes.
4. **Documentation**: update `README.md` and any relevant files in `docs/` or `tutorial/` to reflect the changes.
