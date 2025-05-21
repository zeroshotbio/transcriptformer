import subprocess
import sys
from pathlib import Path

from test.utils_data import make_fake_anndata


def test_cli_dry_run(tmp_path: Path) -> None:
    ad_path = tmp_path / "toy.h5ad"
    make_fake_anndata(n_cells=4, n_genes=32).write(ad_path)

    cmd = [
        sys.executable,
        "-m",
        "transcriptformer.finetune_lora",
        "--checkpoint-path",
        "test/assets/mini_ckpt.pt",
        "--train-files",
        str(ad_path),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--lora-r",
        "2",
        "--lora-alpha",
        "4",
        "--devices",
        "cpu",
        "--output-path",
        str(tmp_path / "adapters.pt"),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    assert proc.returncode == 0, proc.stderr
    assert "Dry-run complete" in proc.stdout
