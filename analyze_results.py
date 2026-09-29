"""Analysis and Quality Verification module for AlphaFold2-multimer predictions.

Functions:
- Inspects and validates Predicted Aligned Error (PAE) matrices
- Computes chain-by-chain and interface pLDDT / ipTM scores
- Generates publication-quality PAE heatmaps with chain demarcations
- Analyzes specific biological contact interfaces:
  * Construct A: Upright trimer MUN interface
  * Construct B: Electrostatic contact patch (K1495/K1500 <-> D1358/D1369)
  * Construct C: Munc13-1 C2A and RIM1alpha zinc-finger heterodimer interface
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_fasta_chain_lengths(fasta_path: Path) -> List[int]:
    """Extract individual chain lengths from a ColabFold ':'-delimited FASTA file."""
    with open(fasta_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    seq_lines = [l for l in lines if not l.startswith(">")]
    full_seq = "".join(seq_lines)
    chains = full_seq.split(":")
    return [len(c) for c in chains]


def analyze_prediction_output(
    target_dir: Path,
    fasta_path: Optional[Path] = None,
) -> Dict[str, any]:
    """Parse scores JSON, plot PAE heatmap, and analyze interface quality."""
    target_dir = Path(target_dir)
    score_files = sorted(target_dir.glob("*_scores_rank_001*.json"))
    if not score_files:
        # Fallback to any scores json
        score_files = sorted(target_dir.glob("*scores*.json"))

    if not score_files:
        print(f"No score JSON files found in {target_dir}")
        return {}

    score_file = score_files[0]
    with open(score_file, "r") as f:
        data = json.load(f)

    plddt = np.array(data.get("plddt", []))
    pae = np.array(data.get("pae", data.get("predicted_aligned_error", [])))
    iptm = float(data.get("iptm", 0.0))
    ptm = float(data.get("ptm", 0.0))
    mean_plddt = float(np.mean(plddt)) if len(plddt) > 0 else 0.0

    # Determine chain lengths
    chain_lengths = []
    if fasta_path and Path(fasta_path).exists():
        chain_lengths = parse_fasta_chain_lengths(Path(fasta_path))
    elif len(pae) > 0:
        # If no fasta provided, check if config.json has it
        config_file = target_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    cfg = json.load(f)
                    seq = cfg.get("sequence", "")
                    if ":" in seq:
                        chain_lengths = [len(c) for c in seq.split(":")]
            except Exception:
                pass

    print("\n" + "=" * 65)
    print(f"Quality Analysis for: {target_dir.name}")
    print(f"Score file: {score_file.name}")
    print(f"Overall Mean pLDDT: {mean_plddt:.2f}")
    print(f"pTM Score:         {ptm:.3f}")
    print(f"ipTM Score:        {iptm:.3f}")
    if chain_lengths:
        print(f"Chains ({len(chain_lengths)}): lengths = {chain_lengths}")
    print("=" * 65)

    # Plot and save PAE matrix if available
    pae_plot_path = None
    if len(pae) > 0:
        pae_plot_path = plot_pae_matrix(
            pae=pae,
            chain_lengths=chain_lengths,
            title=f"Predicted Aligned Error (PAE) - {target_dir.name}\nMean pLDDT: {mean_plddt:.1f} | ipTM: {iptm:.2f}",
            output_path=target_dir / f"{target_dir.name}_pae_annotated.png",
        )

    # Biological interface checks
    interface_report = evaluate_biological_interfaces(
        target_name=target_dir.name,
        pae=pae,
        plddt=plddt,
        chain_lengths=chain_lengths,
    )

    return {
        "target_name": target_dir.name,
        "mean_plddt": mean_plddt,
        "ptm": ptm,
        "iptm": iptm,
        "chain_lengths": chain_lengths,
        "pae_plot": str(pae_plot_path) if pae_plot_path else None,
        "interface_report": interface_report,
    }


def plot_pae_matrix(
    pae: np.ndarray,
    chain_lengths: List[int],
    title: str,
    output_path: Path,
) -> Path:
    """Generate annotated PAE heatmap with chain boundaries and colorbar."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    im = ax.imshow(pae, cmap="bwr_r", vmin=0, vmax=30, origin="upper")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Expected position error (Å)", rotation=270, labelpad=15, fontsize=11)

    # Annotate chain boundaries
    if chain_lengths and len(chain_lengths) > 1:
        cum_lengths = np.cumsum(chain_lengths)
        chain_letters = [chr(ord("A") + i) for i in range(len(chain_lengths))]

        # Midpoints for tick labels
        midpoints = [cum_lengths[0] / 2] + [
            (cum_lengths[i - 1] + cum_lengths[i]) / 2 for i in range(1, len(cum_lengths))
        ]

        # Draw grid lines separating chains
        for bound in cum_lengths[:-1]:
            ax.axvline(bound - 0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.axhline(bound - 0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.8)

        ax.set_xticks(midpoints)
        ax.set_xticklabels([f"Chain {l}" for l in chain_letters], fontsize=10, fontweight="bold")
        ax.set_yticks(midpoints)
        ax.set_yticklabels([f"Chain {l}" for l in chain_letters], fontsize=10, fontweight="bold")
    else:
        ax.set_xlabel("Scored Residue", fontsize=11)
        ax.set_ylabel("Aligned Residue", fontsize=11)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved annotated PAE plot to: {output_path}")
    return output_path


def evaluate_biological_interfaces(
    target_name: str,
    pae: np.ndarray,
    plddt: np.ndarray,
    chain_lengths: List[int],
) -> Dict[str, any]:
    """Specific interface evaluation based on construct context."""
    report = {}
    if len(pae) == 0 or len(chain_lengths) < 2:
        return report

    cum = np.cumsum([0] + chain_lengths)

    # Helper to get inter-chain sub-matrix
    def get_interchain_pae(chain_i: int, chain_j: int) -> np.ndarray:
        return pae[cum[chain_i]:cum[chain_i + 1], cum[chain_j]:cum[chain_j + 1]]

    if "construct_b" in target_name.lower():
        # Construct B: Electrostatic contact patch (D1358/D1369 and K1495/K1500)
        # Monomer is 569 AA (residues 1148-1735 in canonical numbering).
        # Offset: residue 1148 corresponds to index 0 of each monomer.
        # D1358 is at 1358 - 1148 = 210
        # D1369 is at 1369 - 1148 = 221
        # K1495 is at 1495 - 1148 = 347
        # K1500 is at 1500 - 1148 = 352
        # (accounting for any small indel shifts if applicable)
        print("\n--- Specific Contact Evaluation: Construct B Electrostatic Trimer Patch ---")
        monomer_len = chain_lengths[0]
        # Locate exact patches around 200-230 and 340-360 in monomer
        d_patch = (max(0, 205), min(monomer_len, 225))
        k_patch = (max(0, 342), min(monomer_len, 357))

        contacts = [
            ("Chain A (K-patch) -> Chain B (D-patch)", 0, 1, k_patch, d_patch),
            ("Chain B (K-patch) -> Chain C (D-patch)", 1, 2, k_patch, d_patch),
            ("Chain C (K-patch) -> Chain A (D-patch)", 2, 0, k_patch, d_patch),
        ]
        patch_errors = []
        for desc, ci, cj, (r_start, r_end), (c_start, c_end) in contacts:
            sub = pae[cum[ci] + r_start : cum[ci] + r_end, cum[cj] + c_start : cum[cj] + c_end]
            min_err = float(np.min(sub)) if sub.size > 0 else 99.0
            mean_err = float(np.mean(sub)) if sub.size > 0 else 99.0
            patch_errors.append(min_err)
            print(f"  {desc}: Min PAE = {min_err:.2f} Å | Mean PAE = {mean_err:.2f} Å")
            report[desc] = {"min_pae": min_err, "mean_pae": mean_err}

        report["mean_trimer_patch_pae"] = float(np.mean(patch_errors))

    elif "construct_a" in target_name.lower():
        # Construct A: Upright trimer MUN interface across 3 subunits
        print("\n--- Specific Interface Evaluation: Construct A Upright Trimer ---")
        trimer_pairs = [(0, 1), (1, 2), (2, 0)]
        pair_means = []
        for ci, cj in trimer_pairs:
            sub = get_interchain_pae(ci, cj)
            min_err = float(np.min(sub))
            mean_err = float(np.mean(sub))
            pair_means.append(mean_err)
            lbl = f"Chain {chr(65+ci)} - Chain {chr(65+cj)}"
            print(f"  {lbl}: Min PAE = {min_err:.2f} Å | Mean PAE = {mean_err:.2f} Å")
            report[lbl] = {"min_pae": min_err, "mean_pae": mean_err}
        report["mean_interchain_pae"] = float(np.mean(pair_means))

    elif "construct_c" in target_name.lower():
        # Construct C: Munc13-1 C2A (Chain A) & RIM1alpha ZF (Chain B)
        print("\n--- Specific Interface Evaluation: Active Zone Condensate Heterodimer ---")
        sub_ab = get_interchain_pae(0, 1)
        sub_ba = get_interchain_pae(1, 0)
        min_ab = float(np.min(sub_ab))
        min_ba = float(np.min(sub_ba))
        print(f"  C2A -> RIM1-ZF Min PAE: {min_ab:.2f} Å")
        print(f"  RIM1-ZF -> C2A Min PAE: {min_ba:.2f} Å")
        report["c2a_rim1_min_pae"] = min_ab
        report["rim1_c2a_min_pae"] = min_ba

    return report


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "outputs/construct_b"
    fasta = sys.argv[2] if len(sys.argv) > 2 else "targets/construct_b.fasta"
    analyze_prediction_output(Path(target), Path(fasta))
