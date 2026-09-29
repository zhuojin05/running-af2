#!/usr/bin/env python3
"""CLI Entrypoint for Munc13-1 AlphaFold2-multimer Predictions on Modal.

Features:
- Validates FASTA headers and multimer syntax (chains delimited by ':')
- Pre-checks and commits AlphaFold2 model weights in Modal Volume ('colabfold-params')
- Dispatches inference to remote 80 GB A100 GPU on Modal
- Streams execution logs in real time
- Synchronizes output PDB coordinates, PAE/pLDDT scores JSON, MSAs, and plots locally
- Automatically runs PAE and interface quality verification (ipTM, trimer contacts)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import modal

from analyze_results import analyze_prediction_output
from app import app, download_model_weights
from runner import predict_structure


# Standard targets mapping
TARGET_MAP = {
    "construct_a": Path("targets/construct_a.fasta"),
    "construct_b": Path("targets/construct_b.fasta"),
    "construct_c": Path("targets/construct_c.fasta"),
}


def validate_multimer_fasta(fasta_path: Path) -> Tuple[str, str, List[int]]:
    """Validate FASTA format, header, and ColabFold multimer chain syntax."""
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    content = fasta_path.read_text().strip()
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Invalid FASTA in {fasta_path}: Missing header starting with '>'")

    header = lines[0][1:].strip()
    seq_lines = lines[1:]
    full_sequence = "".join(seq_lines).replace(" ", "").upper()

    # Verify multimer delimiter syntax
    chains = full_sequence.split(":")
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    for idx, chain in enumerate(chains, 1):
        if not chain:
            raise ValueError(f"Empty chain found at position {idx} in {fasta_path}")
        invalid = set(chain) - valid_aa
        if invalid:
            raise ValueError(
                f"Invalid amino acid characters {invalid} in chain {idx} of {fasta_path}"
            )

    chain_lengths = [len(c) for c in chains]
    return header, full_sequence, chain_lengths


def ensure_cached_weights():
    """Verify or trigger pre-download of weights to persistent Modal volume."""
    print("\n[Modal Volume Check] Verifying AlphaFold2 model weights in 'colabfold-params'...")
    with app.run():
        result = download_model_weights.remote(force=False)
        print(f"[Modal Volume Check] Status: {result.get('status')}")
        print(f"[Modal Volume Check] File count in cache: {result.get('file_count')}")


def run_single_prediction(
    fasta_path: Path,
    output_base_dir: Path,
    num_recycle: int = 3,
    use_gpu_relax: bool = True,
    dry_run: bool = False,
) -> bool:
    """Orchestrate single target prediction and synchronize outputs."""
    target_name = fasta_path.stem
    header, sequence, chain_lengths = validate_multimer_fasta(fasta_path)

    print("\n" + "=" * 70)
    print(f"Target:        {target_name}")
    print(f"FASTA File:    {fasta_path}")
    print(f"Header:        >{header}")
    print(f"Chains ({len(chain_lengths)}):   Lengths = {chain_lengths} (Total = {sum(chain_lengths)} AA)")
    print(f"Recycles:      {num_recycle}")
    print(f"GPU Relax:     {use_gpu_relax}")
    print("=" * 70)

    if dry_run:
        print("[DRY-RUN] Validation successful. Skipping remote GPU execution.")
        return True

    target_out_dir = output_base_dir / target_name
    target_out_dir.mkdir(parents=True, exist_ok=True)

    fasta_content = f">{header}\n{sequence}\n"

    print(f"\nDispatching '{target_name}' to Modal (80 GB A100 GPU)...")
    with app.run():
        try:
            prediction_result = predict_structure.remote(
                fasta_content=fasta_content,
                target_name=target_name,
                num_recycle=num_recycle,
                use_gpu_relax=use_gpu_relax,
            )
        except Exception as e:
            print(f"\n[ERROR] Remote execution failed for {target_name}: {e}")
            return False

    # Synchronize returned files locally
    files_map = prediction_result.get("files", {})
    summary = prediction_result.get("summary", {})

    print(f"\nSynchronizing {len(files_map)} output files to: {target_out_dir}")
    for filename, file_bytes in files_map.items():
        dest = target_out_dir / filename
        dest.write_bytes(file_bytes)
        print(f"  Saved: {dest.name} ({len(file_bytes):,} bytes)")

    # Execute post-prediction analysis and PAE plotting
    print("\nRunning post-prediction analysis and quality checks...")
    try:
        analysis_report = analyze_prediction_output(target_out_dir, fasta_path)
        if analysis_report:
            import json
            metrics_file = target_out_dir / "summary_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(analysis_report, f, indent=2)
            print(f"Saved summary metrics to: {metrics_file}")
    except Exception as e:
        print(f"Warning during post-prediction analysis: {e}")

    print(f"\n[SUCCESS] Completed workflow for target: {target_name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Serverless Munc13-1 AlphaFold2-multimer Prediction Pipeline on Modal"
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default=None,
        help="Target to predict: 'construct_a', 'construct_b', 'construct_c', or path to .fasta file (default: construct_b)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three standard constructs (A, B, and C) sequentially",
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Explicitly download and commit AlphaFold2 model weights to the Modal volume",
    )
    parser.add_argument(
        "--num-recycle",
        type=int,
        default=3,
        help="Number of AlphaFold recycling iterations (default: 3)",
    )
    parser.add_argument(
        "--no-relax",
        action="store_true",
        help="Disable Amber GPU relaxation (relaxation is enabled by default)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("outputs"),
        help="Local directory to store synchronized outputs (default: outputs/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and environment without launching GPU compute",
    )

    args = parser.parse_args()

    # 1. Pre-download weights if explicitly requested
    if args.download_weights:
        ensure_cached_weights()
        if not args.all and args.target is None:
            return

    # 2. Resolve targets
    target_key = args.target if args.target is not None else "construct_b"
    targets_to_run: List[Path] = []
    if args.all:
        targets_to_run = [TARGET_MAP["construct_a"], TARGET_MAP["construct_b"], TARGET_MAP["construct_c"]]
    else:
        key = target_key.lower().strip()
        if key in TARGET_MAP:
            targets_to_run = [TARGET_MAP[key]]
        else:
            p = Path(target_key)
            if not p.exists():
                print(f"Error: Unknown target key '{target_key}' and file not found.")
                sys.exit(1)
            targets_to_run = [p]

    # 3. Execute
    success_count = 0
    for fasta_file in targets_to_run:
        ok = run_single_prediction(
            fasta_path=fasta_file,
            output_base_dir=args.output_dir,
            num_recycle=args.num_recycle,
            use_gpu_relax=not args.no_relax,
            dry_run=args.dry_run,
        )
        if ok:
            success_count += 1

    print("\n" + "=" * 70)
    print(f"Pipeline finished: {success_count}/{len(targets_to_run)} targets succeeded.")
    print("=" * 70)


if __name__ == "__main__":
    main()
