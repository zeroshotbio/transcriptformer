# TranscriptFormer

<p align="center">
  <img src="assets/model_overview.png" width="600" alt="TranscriptFormer Overview">
  <br>
  <em>Overview of TranscriptFormer pretraining data (A), model (B), outputs (C) and downstream tasks (D).
</em>
</p>

**Authors:** James D Pearce, Sara E Simmonds*, Gita Mahmoudabadi*, Lakshmi Krishnan*, Giovanni
Palla, Ana-Maria Istrate, Alexander Tarashansky, Benjamin Nelson, Omar Valenzuela,
Donghui Li, Stephen R Quake, Theofanis Karaletsos (Chan Zuckerberg Initiative)

*Equal contribution

## Description

TranscriptFormer is a family of generative foundation models representing a cross-species generative cell atlas trained on up to 112 million cells spanning 1.53 billion years of evolution across 12 species. The models include three distinct versions:

- **TF-Metazoa**: Trained on 112 million cells spanning all twelve species. The set covers six vertebrates (human, mouse, rabbit, chicken, African clawed frog, zebrafish), four invertebrates (sea urchin, C. elegans, fruit fly, freshwater sponge), plus a fungus (yeast) and a protist (malaria parasite).
The model includes 444 million trainable parameters and 633 million non-trainable
parameters (from frozen pretrained embeddings). Vocabulary size: 247,388.

- **TF-Exemplar**: Trained on 110 million cells from human and four model organisms: mouse (M. musculus), zebrafish (D. rerio), fruit fly (D. melanogaster ), and C. ele-
gans. Total trainable parameters: 542 million; non-trainable: 282 million. Vocabulary size:
110,290.

- **TF-Sapiens**: Trained on 57 million human-only cells. This model has 368 million trainable parameters and 61 million non-trainable parameters. Vocabulary size: 23,829.


TranscriptFormer is designed to learn rich, context-aware representations of single-cell transcriptomes while jointly modeling genes and transcripts using a novel generative architecture. It employs a generative autoregressive joint model over genes and their expression levels per cell across species, with a transformer-based architecture, including a novel coupling between gene and transcript heads, expression-aware multi-head self-attention, causal masking, and a count likelihood to capture transcript-level variability. TranscriptFormer demonstrates robust zero-shot performance for cell type classification across species, disease state identification in human cells, and prediction of cell type specific transcription factors and gene-gene regulatory relationships. This work establishes a powerful framework for integrating and interrogating cellular diversity across species as well as offering a foundation for in-silico experimentation with a generative single-cell atlas model.

For more details, please refer to our manuscript: [A Cross-Species Generative Cell Atlas Across 1.5 Billion Years of Evolution: The TranscriptFormer Single-cell Model](https://www.biorxiv.org/content/10.1101/2025.04.25.650731v1)



## Installation

Transcriptformer requires Python >=3.11.

### Install from PyPI

```bash
# Create and activate a virtual environment
uv venv --python=3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from PyPI
uv pip install transcriptformer
```

### Install from source

```bash
# Clone the repository
git clone https://github.com/czi-ai/transcriptformer.git
cd transcriptformer

# Create and activate a virtual environment with Python 3.11
uv venv --python=3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e .
```

### Requirements

Transcriptformer has the following core dependencies:
- PyTorch (<=2.5.1, as 2.6.0+ may cause pickle errors)
- PyTorch Lightning
- anndata
- scanpy
- numpy
- pandas
- h5py
- hydra-core

See the `pyproject.toml` file for the complete list of dependencies.

### Hardware Requirements
- GPU (A100 40GB recommended) for efficient inference and embedding extraction.
- Can also use a GPU with a lower amount of VRAM (16GB) by setting the inference batch size to 1-4.


## Using the TranscriptFormer CLI

After installing the package, you'll have access to the `transcriptformer` command-line interface (CLI), which provides easy access to download model artifacts and run inference.

### Downloading Model Weights

Use the CLI to download model weights and artifacts from AWS S3:

```bash
# Download a specific model
transcriptformer download tf-sapiens
transcriptformer download tf-exemplar
transcriptformer download tf-metazoa

# Download all models and embeddings
transcriptformer download all

# Download only the embedding files
transcriptformer download all-embeddings

# Specify a custom checkpoint directory
transcriptformer download tf-sapiens --checkpoint-dir /path/to/custom/dir
```

The command will download and extract the following files to the `./checkpoints` directory (or your specified directory):
- `./checkpoints/tf_sapiens/`: Sapiens model weights
- `./checkpoints/tf_exemplar/`: Exemplar model weights
- `./checkpoints/tf_metazoa/`: Metazoa model weights
- `./checkpoints/all_embeddings/`: Embedding files for out-of-distribution species

### Running Inference

Use the CLI to run inference with TranscriptFormer models:

```bash
# Basic inference on in-distribution species (e.g., human with TF-Sapiens)
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/human_val.h5ad \
  --output-path ./inference_results \
  --batch-size 8

# Inference on out-of-distribution species (e.g., mouse with TF-Sapiens)
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/mouse_val.h5ad \
  --pretrained-embedding ./checkpoints/all_embeddings/mus_musculus_gene.h5 \
  --batch-size 8
```

### Advanced Configuration

For advanced configuration options not exposed as CLI arguments, use the `--config-override` parameter:

```bash
transcriptformer inference \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --data-file test/data/human_val.h5ad \
  --config-override model.data_config.normalize_to_scale=10000 \
  --config-override model.inference_config.obs_keys.0=cell_type
```

To see all available CLI options:

```bash
transcriptformer inference --help
transcriptformer download --help
```

### Input Data Format and Preprocessing:

Input data files should be in H5AD format (AnnData objects) with the following requirements:

- **Gene IDs**: The `var` dataframe must contain an `ensembl_id` column with Ensembl gene identifiers
  - Out-of-vocabulary gene IDs will be automatically filtered out during processing
  - Only genes present in the model's vocabulary will be used for inference
  - The column name can be changed using `model.data_config.gene_col_name`

- **Expression Data**: The model expects unnormalized count data and will look for it in the following order:
  1. `adata.raw.X` (if available)
  2. `adata.X`

  This behavior can be controlled using `model.data_config.use_raw`:
  - `None` (default): Try `adata.raw.X` first, then fall back to `adata.X`
  - `True`: Use only `adata.raw.X`
  - `False`: Use only `adata.X`

- **Count Processing**:
  - Count values are clipped at 30 by default (as was done in training)
  - If this seems too low, you can either:
    1. Use `model.data_config.normalize_to_scale` to scale total counts to a specific value (e.g., 1e3-1e4)
    2. Increase `model.data_config.clip_counts` to a value > 30

- **Cell Metadata**: Any cell metadata in the `obs` dataframe will be preserved in the output

No other data preprocessing is necessary - the model handles all required transformations internally. You do not need to perform any additional normalization, scaling, or transformation of the count data before input.

### Output Format:

The inference results will be saved to the specified output directory (default: `./inference_results`) in a file named `embeddings.h5ad`. This is an AnnData object where:

- Cell embeddings are stored in `obsm['embeddings']`
- Original cell metadata is preserved in the `obs` dataframe
- Log-likelihood scores (if available) are stored in `uns['llh']`

For detailed configuration options, see the `src/transcriptformer/cli/conf/inference_config.yaml` file.

## Fine-tuning with LoRA

This repository includes a lightweight [LoRA](https://arxiv.org/abs/2106.09685) implementation for adapting TranscriptFormer to new datasets.



Run the example with:

```bash
python finetune_lora.py \
  --checkpoint-path ./checkpoints/tf_sapiens \
  --train-files path/to/train.h5ad \
  --lora-r 4 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --lora-target-modules linear1 linear2 linears
```

The script uses `AnnDataset` for preprocessing so the inputs match those used during pretraining. Training logic is minimal (one epoch with PyTorch Lightning) and is intended as a starting point for custom fine‑tuning workflows. Only the adapter weights are saved to keep checkpoints small.

## Contributing
This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

## Reporting Security Issues
Please note: If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.

## Citation

If you use TranscriptFormer in your research, please cite:
Pearce, J. D., et. al. (2025). A Cross-Species Generative Cell Atlas Across 1.5 Billion Years of Evolution: The TranscriptFormer Single-cell Model. bioRxiv. Retrieved April 29, 2025, from https://www.biorxiv.org/content/10.1101/2025.04.25.650731v1


Below is a concise “Windows + CUDA quick-start” block you can drop straight into the Installation / Quick-start section of README.md. After adding it, commit and push the change exactly as you do for any other file.

markdown
Copy
Edit
## Windows (CUDA) quick-start 🪟⚡

PyTorch 2.5 turns **torch.compile** on by default and the Flex-Attention layer
in TranscriptFormer likewise tries to JIT-compile its masks.
Because Triton does not (yet) ship official Windows wheels, the first
inference run may abort with:

RuntimeError: Cannot find a working triton installation …

shell
Copy
Edit

Until Triton for Windows lands, disable the compile stack and run in eager
CUDA:

```powershell
# PowerShell – place these three lines in  .venv\Scripts\activate.d\no_compile.ps1
$env:TORCH_COMPILE                           = '0'   # global: skip torch.compile
$env:TORCHDYNAMO_DISABLE                     = '1'   # belt-and-suspenders
$env:TORCH_NN_FLEX_ATTENTION_DISABLE_COMPILE = '1'   # flex-attention specific
Open a new shell, activate the venv, reinstall the CUDA wheels if needed:

powershell
Copy
Edit
pip install --force-reinstall ^
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 ^
  --extra-index-url https://download.pytorch.org/whl/cu121
Then the usual smoke test works on an RTX-class GPU:

powershell
Copy
Edit
transcriptformer download tf-sapiens --checkpoint-dir .\checkpoints
transcriptformer inference `
    --checkpoint-path .\checkpoints\tf_sapiens `
    --data-file      test\data\human_val.h5ad `
    --output-path    .\inference_smoke `
    --batch-size     2
If you do install an unofficial Triton wheel
(pip install triton==2.2.0), simply delete the three environment variables
and reopen your shell to re-enable full JIT.
