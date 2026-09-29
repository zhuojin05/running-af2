"""Modal Inference Runner for Munc13-1 AlphaFold2-multimer complexes.

Configures remote Modal function with:
- 80 GB A100 GPU (modal.gpu.A100(size="80GB") / "A100-80GB")
- TF_FORCE_UNIFIED_MEMORY="1" and XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
- ColabFold batch execution with --model-type alphafold2_multimer_v3,
  --num-recycle 3, --model-order 1, --use-gpu-relax, and persistent cached weights.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict
import modal

from app import (
    app,
    colabfold_image,
    params_volume,
    PARAMS_DIR,
    outputs_volume,
    OUTPUTS_DIR,
)

# ---------------------------------------------------------------------------
# GPU Configuration
# In Modal 1.x, "A100-80GB" specifies the 80 GB A100 GPU requested in spec
# ---------------------------------------------------------------------------
GPU_A100_80GB = "A100-80GB"

# JAX memory configuration for native 80 GB A100 VRAM utilization (disables unified memory thrashing)
EXECUTION_ENV = {
    "TF_FORCE_UNIFIED_MEMORY": "0",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    "PYTHONUNBUFFERED": "1",
    "COLABFOLDDIR": PARAMS_DIR,
}


@app.function(
    image=colabfold_image,
    gpu=GPU_A100_80GB,
    volumes={
        PARAMS_DIR: params_volume,
        OUTPUTS_DIR: outputs_volume,
    },
    env=EXECUTION_ENV,
    timeout=14400,  # 4 hours per prediction
)
def predict_structure(
    fasta_content: str,
    target_name: str,
    num_recycle: int = 3,
    use_gpu_relax: bool = True,
    model_order: int = 1,
) -> Dict[str, Any]:
    """Execute remote AlphaFold2-multimer prediction on an 80 GB A100 GPU.

    Args:
        fasta_content: Full FASTA text including header and ':'-delimited multimer chains.
        target_name: Name of the construct (e.g. construct_a, construct_b, construct_c).
        num_recycle: Number of recycles (default: 3).
        use_gpu_relax: Whether to use Amber GPU relaxation (default: True).
        model_order: Model order index (default: 1).

    Returns:
        Dictionary containing execution summary metrics, logs, and a mapping of
        output filenames to raw bytes for local synchronization.
    """
    start_time = time.time()
    print("=" * 70)
    print(f"Starting AlphaFold2-multimer inference for target: {target_name}")
    print(f"GPU requested: {GPU_A100_80GB}")
    print(f"Unified memory: TF_FORCE_UNIFIED_MEMORY={os.environ.get('TF_FORCE_UNIFIED_MEMORY')}")
    print(f"XLA memory fraction: XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}")
    print("=" * 70)

    # 1. Verify that model weights are present in persistent volume
    params_path = Path(PARAMS_DIR)
    marker = params_path / "download_finished.txt"
    if not marker.exists():
        print(f"Model parameters missing in {PARAMS_DIR}. Running automatic download...")
        params_path.mkdir(parents=True, exist_ok=True)
        py_dl = (
            "from pathlib import Path; "
            "from colabfold.download import download_alphafold_params; "
            f"download_alphafold_params('alphafold2_multimer_v3', Path('{PARAMS_DIR}')); "
            f"download_alphafold_params('alphafold2_ptm', Path('{PARAMS_DIR}'))"
        )
        dl_res = subprocess.run(["python3", "-c", py_dl], text=True)
        if dl_res.returncode != 0:
            for m_type in ["alphafold2_multimer_v3", "alphafold2_ptm"]:
                subprocess.run(["python3", "-m", "colabfold.download", m_type, str(params_path)], check=True)
        marker.write_text("download_finished\n")
        params_volume.commit()
        print("Committed downloaded weights to persistent volume.")

    # 2. Stage FASTA target input
    input_dir = Path(f"/tmp/colabfold_input_{target_name}")
    input_dir.mkdir(parents=True, exist_ok=True)
    fasta_file = input_dir / f"{target_name}.fasta"
    fasta_file.write_text(fasta_content.strip() + "\n")
    print(f"Staged FASTA to {fasta_file}:")
    for line in fasta_content.strip().splitlines()[:4]:
        print(f"  {line[:90]}...")

    # 3. Prepare output directory
    work_out = Path(f"/tmp/colabfold_output_{target_name}")
    if work_out.exists():
        shutil.rmtree(work_out)
    work_out.mkdir(parents=True, exist_ok=True)

    # 4. Construct colabfold_batch invocation command
    cmd = [
        "colabfold_batch",
        str(fasta_file),
        str(work_out),
        "--model-type", "alphafold2_multimer_v3",
        "--num-recycle", str(num_recycle),
        "--num-models", "1",
        "--model-order", str(model_order),
        "--data", PARAMS_DIR,
    ]
    if use_gpu_relax:
        cmd.append("--use-gpu-relax")

    print(f"\nExecuting command:\n  {' '.join(cmd)}\n")

    # 5. Run prediction with live streaming logs
    sub_env = os.environ.copy()
    sub_env["TF_FORCE_UNIFIED_MEMORY"] = "0"
    sub_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    sub_env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    sub_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=sub_env,
    )
    captured_logs = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
        captured_logs.append(line)
    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        error_msg = f"colabfold_batch failed with exit code {return_code}"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg)

    elapsed_seconds = round(time.time() - start_time, 2)
    print(f"\nPrediction completed in {elapsed_seconds}s. Packaging outputs...")

    # 6. Synchronize output to cloud persistent volume
    cloud_dest = Path(OUTPUTS_DIR) / target_name
    cloud_dest.mkdir(parents=True, exist_ok=True)

    # 7. Collect and bundle output files
    file_bytes_map = {}
    summary_metrics = {
        "target_name": target_name,
        "elapsed_seconds": elapsed_seconds,
        "plddt": None,
        "ptm": None,
        "iptm": None,
        "rank_001_pdb": None,
    }

    for file_path in work_out.glob("*"):
        if file_path.is_file():
            content = file_path.read_bytes()
            file_bytes_map[file_path.name] = content

            # Copy to persistent volume
            dest_file = cloud_dest / file_path.name
            dest_file.write_bytes(content)

            # Parse top rank scores JSON if present
            if "scores_rank_001" in file_path.name and file_path.suffix == ".json":
                try:
                    scores_data = json.loads(content.decode("utf-8"))
                    plddt_list = scores_data.get("plddt", [])
                    mean_plddt = sum(plddt_list) / len(plddt_list) if plddt_list else None
                    summary_metrics["plddt"] = round(mean_plddt, 2) if mean_plddt else None
                    summary_metrics["ptm"] = round(scores_data.get("ptm", 0.0), 3)
                    summary_metrics["iptm"] = round(scores_data.get("iptm", 0.0), 3)
                except Exception as e:
                    print(f"Warning: could not parse scores JSON {file_path.name}: {e}")

            if "rank_001" in file_path.name and file_path.suffix == ".pdb":
                summary_metrics["rank_001_pdb"] = file_path.name

    # Commit the cloud outputs volume
    try:
        outputs_volume.commit()
        print(f"Committed {len(file_bytes_map)} files to Modal Volume 'colabfold-outputs'.")
    except Exception as e:
        print(f"Warning: Could not commit outputs volume: {e}")

    print("=" * 70)
    print(f"Summary for {target_name}:")
    print(f"  Elapsed: {elapsed_seconds}s")
    print(f"  Mean pLDDT: {summary_metrics['plddt']}")
    print(f"  pTM: {summary_metrics['ptm']}")
    print(f"  ipTM: {summary_metrics['iptm']}")
    print(f"  Top PDB: {summary_metrics['rank_001_pdb']}")
    print(f"  Total files generated: {len(file_bytes_map)}")
    print("=" * 70)

    return {
        "target_name": target_name,
        "summary": summary_metrics,
        "files": file_bytes_map,
        "logs": "".join(captured_logs),
    }


if __name__ == "__main__":
    print("runner.py loaded successfully.")
    print("Remote function 'predict_structure' configured for Modal execution.")
