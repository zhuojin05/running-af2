"""Modal App Configuration for Munc13-1 AlphaFold2 Multimer Prediction.

Configures:
- Debian-slim base image with ColabFold, Jax (CUDA 12), OpenBabel, and Kalign
- Persistent Modal Volume for caching ~5 GB AlphaFold2 model weights
- Weight download and volume commit routine
"""

import os
import subprocess
from pathlib import Path
import modal

# ---------------------------------------------------------------------------
# 1. Modal App Definition
# ---------------------------------------------------------------------------
app = modal.App("munc13-alphafold")

# ---------------------------------------------------------------------------
# 2. Persistent Storage for AlphaFold2 Model Weights (~5 GB)
# ---------------------------------------------------------------------------
PARAMS_DIR = "/root/params"
params_volume = modal.Volume.from_name("colabfold-params", create_if_missing=True)

# Also create an outputs volume for cloud persistence of predictions
outputs_volume = modal.Volume.from_name("colabfold-outputs", create_if_missing=True)
OUTPUTS_DIR = "/root/outputs"

# ---------------------------------------------------------------------------
# 3. Base Container Image Definition
# ---------------------------------------------------------------------------
# Equipped with:
# - Debian-slim base
# - OpenBabel and Kalign (installed via apt)
# - colabfold[alphafold] and Jax with CUDA 12 support (installed via pip)
colabfold_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "kalign",
        "openbabel",
        "python3-openbabel",
        "git",
        "wget",
        "curl",
        "tar",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
    )
    .pip_install(
        "colabfold[alphafold]==1.6.3",
        "jax[cuda12]",
        extra_index_url="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    )
    .env({
        "COLABFOLDDIR": PARAMS_DIR,
        "TF_FORCE_UNIFIED_MEMORY": "0",
    })
    .add_local_python_source("app", "runner", "analyze_results")
)

# ---------------------------------------------------------------------------
# 4. Weight Download & Volume Commit Function
# ---------------------------------------------------------------------------
@app.function(
    image=colabfold_image,
    volumes={PARAMS_DIR: params_volume},
    timeout=1800,  # 30-minute timeout for downloading ~5 GB
)
def download_model_weights(force: bool = False) -> dict:
    """Download AlphaFold2-multimer model weights into persistent Modal Volume.
    
    Validates presence of weights and marker file. If missing or `force=True`,
    downloads parameters and commits the volume to permanently cache weights
    and eliminate cold-start latency.
    """
    params_path = Path(PARAMS_DIR)
    marker_file = params_path / "download_finished.txt"

    if marker_file.exists() and not force:
        existing_files = [f.name for f in params_path.glob("*")]
        print(f"Weights already present in {PARAMS_DIR} ({len(existing_files)} items).")
        return {
            "status": "already_cached",
            "file_count": len(existing_files),
            "directory": str(params_path),
            "files": existing_files[:10],
        }

    print(f"Starting AlphaFold2 model weights download to {PARAMS_DIR}...")
    params_path.mkdir(parents=True, exist_ok=True)

    # Download weights using colabfold.download with model_type specified first
    print(f"Downloading AlphaFold2 multimer v3 and ptm weights to {PARAMS_DIR}...")
    py_cmd = (
        "from pathlib import Path; "
        "from colabfold.download import download_alphafold_params; "
        f"download_alphafold_params('alphafold2_multimer_v3', Path('{PARAMS_DIR}')); "
        f"download_alphafold_params('alphafold2_ptm', Path('{PARAMS_DIR}'))"
    )
    res = subprocess.run(["python3", "-c", py_cmd], text=True)
    if res.returncode != 0:
        # Secondary fallback: try CLI with model_type argument first
        print("Python API returned error. Trying CLI fallback with explicit model types...")
        for m_type in ["alphafold2_multimer_v3", "alphafold2_ptm"]:
            cli_res = subprocess.run(
                ["python3", "-m", "colabfold.download", m_type, str(params_path)],
                text=True,
            )
            if cli_res.returncode != 0:
                raise RuntimeError(f"Failed to download AlphaFold parameters for {m_type}")

    # Ensure download marker exists
    if not marker_file.exists():
        marker_file.write_text("download_finished\n")

    # Commit persistent volume
    print("Committing model weights to persistent Modal volume 'colabfold-params'...")
    params_volume.commit()
    print("Volume committed successfully. Weights are now permanently cached across executions.")

    committed_files = [f.name for f in params_path.glob("*")]
    return {
        "status": "downloaded_and_committed",
        "file_count": len(committed_files),
        "directory": str(params_path),
        "files": committed_files,
    }


if __name__ == "__main__":
    # Test local syntax
    print("app.py configured successfully:")
    print("  App:", app.name)
    print("  Params Volume:", params_volume)
    print("  Image:", colabfold_image)
