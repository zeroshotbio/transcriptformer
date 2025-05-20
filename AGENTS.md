# TranscriptFormer – Contributor / Codex Guide

Welcome, Codex 👋 This file explains **where to work, how to validate changes, and the style rules** for this repo so that you can ship production‑ready pull‑requests on the first try.

---

## 1  Repository tour

| Path                    | What lives here                                    | Typical changes                             |
| ----------------------- | -------------------------------------------------- | ------------------------------------------- |
| `src/transcriptformer/` | Core library code (model, dataloaders, CLI)        | new features, bug‑fixes, refactors          |
| `conf/`                 | Hydra configuration files for training & inference | add new configs, tweak defaults             |
| `test/`                 | Unit tests and small integration tests             | add / update tests whenever code changes    |
| `notebooks/`            | Exploratory notebooks (excluded from lint)         | **do not** rely on notebooks for validation |
| `assets/`, `README.md`  | Diagrams & documentation                           | keep in sync with API changes               |

> **Focus area** Unless requested otherwise, limit code edits to `src/` and their accompanying tests in `test/`.

---

## 2  Local development environment

The CI and `codex` setup script install everything with:

```bash
# inside fresh container
python -m pip install -U pip wheel
python -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
pre-commit install --install-hooks
```

Environment variables always set:

```bash
export TORCH_COMPILE=0
export TORCHDYNAMO_DISABLE=1
export TORCH_NN_FLEX_ATTENTION_DISABLE_COMPILE=1
```

Codex can assume these wheels & vars are present.

---

## 3  Validation checklist

Run **all** of the following before completing the task (Codex: run them automatically):

| Step                                              | Command                        |
| ------------------------------------------------- | ------------------------------ |
| Unit tests                                        | `pytest -q`                    |
| Lint + type upgrades                              | `ruff check src test`          |
| Auto‑format                                       | `ruff format --check src test` |
| Pre‑commit (aggregates hooks incl. pyproject‑fmt) | `pre-commit run --all-files`   |

Pull‑requests **must be green** on every step.

---

## 4  Code style & conventions

* **Line length:** 120 chars (`[tool.ruff] line-length = 120`).
* **Import order:** managed by Ruff (`I` rules).
* **Docstrings:** NumPy style; public functions / classes **must** have examples if behaviour is non‑trivial.
* **Typing:** Use explicit type annotations on all new public symbols.
* **Torch:** CPU wheels only in CI; if you need GPU guards → wrap in `if torch.cuda.is_available():`.
* **Hydra configs:** Keep new `.yaml` files in `conf/`; reference them in docs.

---

## 5  Pull‑request expectations

* **Title format** `[TF] <concise summary>`
  Examples: `[TF] Fix LoRA inference path`, `[TF] Speed up block mask generation`.
* First paragraph: 1‑2 sentences explaining *why* not just *what*.
* Bullet list of major changes.
* `Closes #<issue>` where applicable.
* Include **before/after** benchmarks if you touched model speed.

Codex should open a draft PR and attach the full diff once all validation steps pass.

---

## 6  Common tasks for Codex

1. **Bug‑fix**  – given a failing test or stack‑trace, locate the root cause in `src/` and add a regression test.
2. **Refactor** – split large modules, extract pure functions, or replace hand‑rolled code with library helpers.
3. **Add feature** – implement the new CLI flag / model head and document it.
4. **Docs**      – sync README & doc‑strings after API changes.

---

Happy hacking 🚀
