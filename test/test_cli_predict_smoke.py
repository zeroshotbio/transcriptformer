import subprocess
import sys
from pathlib import Path
from test.utils_data import make_fake_anndata

import numpy as np


def test_cli_predict_smoke(tmp_path: Path) -> None:
    ad_path = tmp_path / "toy.h5ad"
    make_fake_anndata(n_cells=4, n_genes=32).write(ad_path)
    out_path = tmp_path / "preds.npy"

    cmd = [
        sys.executable,
        "-m",
        "transcriptformer.predict",
        "--checkpoint-path",
        "test/assets/mini_ckpt.pt",
        "--input-files",
        str(ad_path),
        "--output-path",
        str(out_path),
        "--batch-size",
        "2",
        "--devices",
        "cpu",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "Dry-run complete" in proc.stdout

    cmd.remove("--dry-run")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert out_path.is_file()
    arr = np.load(out_path)
    assert arr.shape == (4, 32)
