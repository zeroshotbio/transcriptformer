Below is an **augmented version of `AGENTS.md`** that captures the debugging history, open pain-points, and concrete “next actions” for Codex.
Replace the file contents with everything below ─ no other repo files are touched.

````md
# TranscriptFormer – Contributor / Codex Guide

Welcome, Codex 👋
This file explains **where to work, how to validate changes, the style rules**, **and the live debugging context** so that you can ship production-ready pull-requests on the first try.

---

## 1  Repository tour

| Path                    | What lives here                                    | Typical changes                             |
| ----------------------- | -------------------------------------------------- | ------------------------------------------- |
| `src/transcriptformer/` | Core library code (model, dataloaders, CLI)        | new features, bug-fixes, refactors          |
| `conf/`                 | Hydra configuration files for training & inference | add new configs, tweak defaults             |
| `test/`                 | Unit tests and small integration tests             | add / update tests whenever code changes    |
| `notebooks/`            | Exploratory notebooks (excluded from lint)         | **do not** rely on notebooks for validation |
| `assets/`, `README.md`  | Diagrams & documentation                           | keep in sync with API changes               |

> **Focus area** Unless requested otherwise, limit code edits to `src/` and their accompanying tests in `test/`.

---

## 2  Local development environment

The CI and `codex` setup script install everything with:

```bash
# inside fresh container
python -m pip install -U pip wheel
python -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
pre-commit install --install-hooks
````

Environment variables always set:

```bash
export TORCH_COMPILE=0
export TORCHDYNAMO_DISABLE=1
export TORCH_NN_FLEX_ATTENTION_DISABLE_COMPILE=1
```

Codex can assume these wheels & vars are present.

---

## 3  Validation checklist

Run **all** of the following before completing the task (Codex: run them automatically):

| Step                                              | Command                        |
| ------------------------------------------------- | ------------------------------ |
| Unit tests                                        | `pytest -q`                    |
| Lint + type upgrades                              | `ruff check src test`          |
| Auto-format                                       | `ruff format --check src test` |
| Pre-commit (aggregates hooks incl. pyproject-fmt) | `pre-commit run --all-files`   |
| CLI smoke test                                    | `pytest test/test_cli_smoke.py::test_cli_dry_run -q` |
| CLI predict smoke test                             | `pytest test/test_cli_predict_smoke.py::test_cli_predict_smoke -q` |

Pull-requests **must be green** on every step.

---

## 4  Code style & conventions

* **Line length:** 120 chars (`[tool.ruff] line-length = 120`).
* **Import order:** managed by Ruff (`I` rules).
* **Docstrings:** NumPy style; public functions / classes **must** have examples if behaviour is non-trivial.
* **Typing:** Use explicit type annotations on all new public symbols.
* **Torch:** CPU wheels only in CI; if you need GPU guards → wrap in `if torch.cuda.is_available():`.
* **Hydra configs:** Keep new `.yaml` files in `conf/`; reference them in docs.

---

## 5  Pull-request expectations

* **Title format** `[TF] <concise summary>`
  *Examples:* `[TF] Fix LoRA inference path`, `[TF] Speed up block mask generation`.
* First paragraph: 1-2 sentences explaining *why* not just *what*.
* Bullet list of major changes.
* `Closes #<issue>` where applicable.
* Include **before/after** benchmarks if you touched model speed.

Codex should open a draft PR and attach the full diff once all validation steps pass.

---

## 6  Common tasks for Codex

1. **Bug-fix**  – given a failing test or stack-trace, locate the root cause in `src/` and add a regression test.
2. **Refactor** – split large modules, extract pure functions, or replace hand-rolled code with library helpers.
3. **Add feature** – implement the new CLI flag / model head and document it.
4. **Docs**      – sync README & doc-strings after API changes.

---

## 7  ⏱️ Live debugging context (May 2025)

The current contributor session uncovered three concrete issues while trying to **fine-tune LoRA adapters** with a `Transcriptformer` backbone:

| ID     | Symptom / Trace                                                                                                  | Root-cause guess                                                                                                                                      | Suggested fix area                                     |
| ------ | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **B1** | `IndexError: index 2047 is out of bounds for dimension 1` inside `pad_mask` → triggered by `flex_attention.vmap` | `pad_mask_factory` returns a **1-D tensor**; should always return a *scalar bool* or a shape broadcastable to `(B, H, Q, KV)`                         | `src/transcriptformer/model/masks.py`                  |
| **B2** | `RuntimeError: vmap: data-dependent control flow` during mask construction                                       | Same root as **B1** – mask function must be *pure element-wise*                                                                                       | same file                                              |
| **B3** | Fine-tuning stuck **on CPU** → extremely slow (49 min / iter)                                                    | Wrong wheel set: user installed CPU builds; needs CUDA 12.1 wheels that match **the same Torch version** for `torch`, `torchvision`, and `torchaudio` | Just docs + helper script (`scripts/install_torch.py`) |

> **Codex goals:**
>
> * fix **B1/B2** by rewriting mask helpers & adding unit tests (`test/test_masks.py`).
> * optional – create `scripts/install_gpu_torch.ps1` that pins a consistent CUDA wheel trio.

---

### 7.1  Reproduction snippet

```bash
python finetune_lora.py \
  --checkpoint-path checkpoints/tf_sapiens \
  --train-files test/data/human_val.h5ad \
  --epochs 1 --batch-size 4 \
  --lora-r 4 --lora-alpha 16 \
  --devices 0 --output-path adapters_epoch1.pt
```

On *fresh main* this crashes with **B1**.

---

### 7.2  Acceptance criteria for the fix

* `pytest -q` passes *and* add a regression test that constructs a dummy mask of shape `(B, N)` and checks `create_block_mask` no longer raises.
* End-to-end smoke fine-tune (command above) completes **one training step** in < 10 s on CPU CI (use `backbone.eval()` + `with torch.no_grad()`) – see existing smoke tests for pattern.

---

## 8  Performance notes

* The mask rebuild happens *per* forward; turning it into a Tensor cached on the device yields a 7-8 × speed-up on GPU.
* Avoid Python `for` inside batched dims; use `torch.vmap` or broadcasting.

---

## 9  GPU install cheat-sheet (for humans)

```powershell
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 `
            torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121
```

**Versions must match!** (`torch == torchvision == torchaudio` *major.minor*).

---

Happy hacking 🚀

```

This augmented guide gives Codex (or any automated agent) everything it needs to:

* see the current failure modes,
* know exactly **where** to patch (`masks.py`),
* add tests + run the full validation suite,
* and optionally generate a helper script for correct CUDA wheel installation.

No other project files were modified.
```
