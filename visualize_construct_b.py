# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "pymol-open-source-whl",
# ]
# ///
"""PyMOL Visualization & Structural Inspection for Munc13-1 Construct B Trimer.

Generates:
1. Multi-chain overview cartoon with electrostatic patches highlighted
2. AlphaFold canonical pLDDT confidence visualization
3. Close-up view of the electrostatic contact regions (K-patch vs D-patch)
4. Saves a complete PyMOL session file (.pse) for interactive inspection
"""

import os
import sys
from pathlib import Path

# Headless OSMesa rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pymol
pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()

from pymol import cmd

def main():
    pdb_path = Path("outputs/construct_b/construct_b_mun_c2c_trimer_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb").resolve()
    out_dir = Path("outputs/construct_b").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdb_path.exists():
        print(f"Error: PDB file not found at {pdb_path}")
        cmd.quit()
        sys.exit(1)

    # 1. Load Structure & Pre-flight verification
    cmd.reinitialize()
    obj_name = "construct_b_trimer"
    cmd.load(str(pdb_path), obj_name)

    atom_count = cmd.count_atoms(obj_name)
    if atom_count == 0:
        print(f"Error: No atoms loaded from {pdb_path}")
        cmd.quit()
        sys.exit(1)
    print(f"Loaded {obj_name}: {atom_count:,} atoms across chains.")

    # General styling
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("depth_cue", 1)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("specular", 0.2)

    # Define Electrostatic Patch Selections
    # D-patch: D1358 / D1369 -> residues 209, 211, 219, 222
    # K-patch: K1495 / K1500 -> residues 347, 348, 353
    cmd.select("d_patch_A", "chain A and resi 209+211+219+222")
    cmd.select("d_patch_B", "chain B and resi 209+211+219+222")
    cmd.select("d_patch_C", "chain C and resi 209+211+219+222")

    cmd.select("k_patch_A", "chain A and resi 347+348+353")
    cmd.select("k_patch_B", "chain B and resi 347+348+353")
    cmd.select("k_patch_C", "chain C and resi 347+348+353")

    cmd.select("all_d_patches", "d_patch_A or d_patch_B or d_patch_C")
    cmd.select("all_k_patches", "k_patch_A or k_patch_B or k_patch_C")

    # -------------------------------------------------------------------------
    # Render 1: Multi-chain Cartoon Overview with Highlighted Patches
    # -------------------------------------------------------------------------
    cmd.hide("everything")
    cmd.show("cartoon", obj_name)
    cmd.color("cyan", "chain A")
    cmd.color("palegreen", "chain B")
    cmd.color("salmon", "chain C")

    # Show patches as sticks
    cmd.show("sticks", "all_d_patches")
    cmd.show("sticks", "all_k_patches")
    cmd.color("firebrick", "all_d_patches and elem C")
    cmd.color("marine", "all_k_patches and elem C")
    cmd.color("atomic", "not elem C")

    cmd.orient()
    chain_view_png = out_dir / "construct_b_pymol_chains.png"
    cmd.png(str(chain_view_png), width=1600, height=1200, dpi=150)
    print(f"Rendered chain overview: {chain_view_png}")

    # -------------------------------------------------------------------------
    # Render 2: Canonical AlphaFold pLDDT Spectrum
    # -------------------------------------------------------------------------
    cmd.set_color("af_very_low", [0xFF / 255.0, 0x7D / 255.0, 0x45 / 255.0])   # orange < 50
    cmd.set_color("af_low",      [0xFF / 255.0, 0xDB / 255.0, 0x13 / 255.0])   # yellow 50-70
    cmd.set_color("af_confident",[0x65 / 255.0, 0xCB / 255.0, 0xF3 / 255.0])   # light blue 70-90
    cmd.set_color("af_very_high", [0x00 / 255.0, 0x53 / 255.0, 0xD6 / 255.0])  # dark blue >= 90

    cmd.color("af_very_low", f"{obj_name} and polymer.protein")
    cmd.color("af_low", f"{obj_name} and polymer.protein and b > 50")
    cmd.color("af_confident", f"{obj_name} and polymer.protein and b > 70")
    cmd.color("af_very_high", f"{obj_name} and polymer.protein and b > 90")

    plddt_view_png = out_dir / "construct_b_pymol_plddt.png"
    cmd.png(str(plddt_view_png), width=1600, height=1200, dpi=150)
    print(f"Rendered pLDDT overview: {plddt_view_png}")

    # -------------------------------------------------------------------------
    # Render 3: Zoom / Close-up on Contact Region
    # -------------------------------------------------------------------------
    # Re-apply chain colors
    cmd.color("cyan", "chain A")
    cmd.color("palegreen", "chain B")
    cmd.color("salmon", "chain C")
    cmd.color("firebrick", "all_d_patches and elem C")
    cmd.color("marine", "all_k_patches and elem C")
    cmd.color("atomic", "not elem C")

    cmd.zoom("all_d_patches or all_k_patches", buffer=8.0)
    zoom_view_png = out_dir / "construct_b_pymol_contact_zoom.png"
    cmd.png(str(zoom_view_png), width=1600, height=1200, dpi=150)
    print(f"Rendered contact patch close-up: {zoom_view_png}")

    # -------------------------------------------------------------------------
    # Save Full PyMOL Session (.pse)
    # -------------------------------------------------------------------------
    cmd.zoom(obj_name)
    session_file = out_dir / "construct_b_session.pse"
    cmd.save(str(session_file))
    print(f"Saved PyMOL session file: {session_file}")

    cmd.quit()
    print("PyMOL rendering workflow completed successfully.")

if __name__ == "__main__":
    main()
