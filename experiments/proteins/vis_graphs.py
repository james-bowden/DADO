"""
vis_graphs.py  –  Visualize protein structure, contact graph, and junction-tree
                  decomposition for each protein test case.

Run from the DADO root directory:
    python experiments/proteins/vis_graphs.py

Outputs per protein (in experiments/proteins/vis_graphs/):
    <obj>.png                       – clean (no titles, legends, colorbars)
    <obj>_titles_legends_cbars.png  – full annotations
Also outputs:
    titles_legends_cbars.png        – standalone strip (same width), goes above rows
"""

import os
import sys
import warnings
import traceback
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─── Environment setup (must be before JAX import) ───────────────────────────
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"
)

cwd = os.getcwd()
if not cwd.endswith("DADO"):
    raise ValueError(f"Working directory {cwd} does not end with 'DADO'")
sys.path.append(cwd)

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import networkx as nx
from Bio.PDB import MMCIFParser

plt.style.use("seaborn-v0_8-whitegrid")

from src.problems.real.oracle import OracleObjective
from src.problems.real.protein_structure import residue_contact_map, chain_to_sequence
from src.decomposition.graphs import GeneralizedGraph, JunctionTree, Tree, DisconnectedGraph

# ─── Constants ────────────────────────────────────────────────────────────────

THRESHOLD = 4.5          # Å contact-distance threshold

COL_NON_OPT        = "#E41A1C"  # red  – designable positions / ball-and-stick atoms
COL_BB             = "#AAAAAA"  # light grey – backbone chain
COL_EDGE_CONTACT   = "#1f78b4"  # dark blue – contact edges (panels 2 & 3)
HULL_BUFFER = 3.0        # Å-scale buffer for convex-hull expansion

BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Font sizes (+2 pt from original)
FS_TITLE     = 18
FS_LEGEND    = 14
FS_SCALE_BAR = 10

# Fixed 3-D view for all proteins / all panels 1-3 (after PCA pre-rotation)
VIEW_ELEV = 30
VIEW_AZIM = -60

# Global colour-scale bounds
GLOBAL_MAX_MEMBERSHIP  = 25   # max # JT nodes any position appears in (phot)
GLOBAL_MAX_CLIQUE_SIZE = 26   # max # positions in any single JT node (phot)

STRUCTURE_PATHS_DICT = {
    "amyloid": "fold_amyloid_model_0.cif",
    "aav":     "fold_aav_model_0.cif",
    "gb1_55":  "fold_gb1_model_0.cif",
    "tdp43":   "fold_tdp43_model_0.cif",
    "ynzc":    "fold_ynzc_bacsu_tsuboyama_2023_2jvd_2025_09_18_01_16_model_0.cif",
    "gcn4":    "fold_gcn4_model_0.cif",
    "phot":    "fold_phot_chlre_model_0.cif",
}

# ─── Amino-acid heavy-atom bond connectivity (PDB atom names, no H) ──────────
# Source: wwPDB Chemical Component Dictionary
AA_BONDS = {
    'G': [('N','CA'),('CA','C'),('C','O')],
    'A': [('N','CA'),('CA','C'),('C','O'),('CA','CB')],
    'V': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG1'),('CB','CG2')],
    'L': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','CD1'),('CG','CD2')],
    'I': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG1'),('CB','CG2'),
          ('CG1','CD1')],
    'P': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),('CG','CD'),
          ('CD','N')],
    'F': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','CD1'),('CG','CD2'),('CD1','CE1'),('CD2','CE2'),
          ('CE1','CZ'),('CE2','CZ')],
    'Y': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','CD1'),('CG','CD2'),('CD1','CE1'),('CD2','CE2'),
          ('CE1','CZ'),('CE2','CZ'),('CZ','OH')],
    'W': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','CD1'),('CG','CD2'),('CD1','NE1'),('NE1','CE2'),
          ('CD2','CE2'),('CD2','CE3'),('CE2','CZ2'),('CE3','CZ3'),
          ('CZ2','CH2'),('CZ3','CH2')],
    'S': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','OG')],
    'T': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','OG1'),('CB','CG2')],
    'C': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','SG')],
    'M': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','SD'),('SD','CE')],
    'D': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','OD1'),('CG','OD2')],
    'N': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','OD1'),('CG','ND2')],
    'E': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','CD'),('CD','OE1'),('CD','OE2')],
    'Q': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','CD'),('CD','OE1'),('CD','NE2')],
    'H': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','ND1'),('CG','CD2'),('ND1','CE1'),('CE1','NE2'),('NE2','CD2')],
    'K': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','CD'),('CD','CE'),('CE','NZ')],
    'R': [('N','CA'),('CA','C'),('C','O'),('CA','CB'),('CB','CG'),
          ('CG','CD'),('CD','NE'),('NE','CZ'),('CZ','NH1'),('CZ','NH2')],
}

# ─── Data loading ─────────────────────────────────────────────────────────────

def parse_cif_atoms(cif_path):
    """
    Parse a CIF file; return per-residue atom data + per-atom dict.

    Returns
    -------
    residue_atoms : list of dicts – keys: 'backbone', 'sidechain', 'ca', 'atom_dict'
    residue_str   : str           – one-letter AA sequence
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_path)
    bio_model = next(structure.get_models())
    chain = next(bio_model.get_chains())
    residues, residue_str = chain_to_sequence(chain)

    residue_atoms = []
    for res in residues:
        bb, sc, ca = [], [], None
        atom_dict = {}
        for atom in res.get_atoms():
            name  = atom.get_name().strip()
            coord = atom.get_vector().get_array().copy()
            atom_dict[name] = coord
            if name in BACKBONE_ATOMS:
                bb.append(coord)
                if name == "CA":
                    ca = coord
            else:
                sc.append(coord)
        residue_atoms.append({
            "backbone":  np.array(bb) if bb else np.empty((0, 3)),
            "sidechain": np.array(sc) if sc else np.empty((0, 3)),
            "ca":        ca,
            "atom_dict": atom_dict,
        })
    return residue_atoms, residue_str


def _compute_pca_rotation(ca_coords):
    """
    Compute a 3×3 rotation matrix that aligns the protein's PCA axes with
    the coordinate axes: PC1 → Z (most variance, vertical),
                         PC2 → X (second variance, horizontal),
                         PC3 → Y (least variance, depth).
    Putting PC1 on the vertical (Z) axis makes best use of the tall plot space.
    """
    if len(ca_coords) < 3:
        return np.eye(3)
    pca = PCA(n_components=3)
    pca.fit(ca_coords)
    # rows of pca.components_: PC1, PC2, PC3 (unit vectors in original space)
    # Build R so that: R @ v = coordinates in new frame
    #   new X = PC2, new Y = PC3 (depth), new Z = PC1 (vertical, most variance)
    R = np.array([
        pca.components_[1],   # new X ← PC2 (horizontal)
        pca.components_[2],   # new Y ← PC3 (depth, least variation)
        pca.components_[0],   # new Z ← PC1 (vertical, most variation)
    ])
    return R


def _apply_rotation_to_residues(residue_atoms, R, center):
    """Apply rotation R (3×3) around center to all atom coords in residue_atoms."""
    def _rot(arr):
        if len(arr) == 0:
            return arr
        return (R @ (arr - center).T).T + center

    result = []
    for res in residue_atoms:
        new_ca = (R @ (res["ca"] - center)) + center if res["ca"] is not None else None
        new_ad = {
            name: (R @ (coord - center)) + center
            for name, coord in res["atom_dict"].items()
        }
        result.append({
            "backbone":  _rot(res["backbone"]),
            "sidechain": _rot(res["sidechain"]),
            "ca":        new_ca,
            "atom_dict": new_ad,
        })
    return result


def _compute_jt_membership(gg, L):
    membership = np.ones(L, dtype=int)
    for sg_idx, sg in enumerate(gg.subgraphs):
        if isinstance(sg, JunctionTree):
            g_inds = gg.subgraph_node_indices[sg_idx]
            for gi in g_inds:
                membership[gi] = 0
            for c in range(sg.n_nodes):
                for li in sg.index_to_nodes[c]:
                    membership[g_inds[li]] += 1
    return membership


def load_protein_data(obj_name, structure_path, threshold=THRESHOLD):
    """Load and PCA-rotate all atom data for one protein."""
    obj = OracleObjective(obj_name=obj_name)
    residue_atoms, residue_str = parse_cif_atoms(structure_path)

    if obj.active_inds is None:
        ind_start = residue_str.index(obj.WT_seq)
        active_cif_indices = list(range(ind_start, ind_start + len(obj.WT_seq)))
    else:
        active_cif_indices = list(obj.active_inds)

    active_cif_set = set(active_cif_indices)
    L = len(active_cif_indices)
    assert L == len(obj.D), f"Length mismatch {L} vs {len(obj.D)}"

    edges, _ = residue_contact_map(
        structure_path, obj.WT_seq,
        edgelist=True, verbose=False,
        active_inds=obj.active_inds,
        threshold=threshold,
    )

    gg = GeneralizedGraph(L, edges, verbose=False)

    # Raw active CA coords (before rotation)
    raw_active_ca = []
    for cif_idx in active_cif_indices:
        ca = residue_atoms[cif_idx]["ca"]
        if ca is None:
            bb = residue_atoms[cif_idx]["backbone"]
            ca = bb.mean(axis=0) if len(bb) > 0 else np.zeros(3)
        raw_active_ca.append(ca)
    raw_active_ca = np.array(raw_active_ca)

    # PCA rotation: align PC1→X, PC2→Z, PC3→Y; use fixed view for all proteins
    R_pca  = _compute_pca_rotation(raw_active_ca)
    center = raw_active_ca.mean(axis=0)

    residue_atoms    = _apply_rotation_to_residues(residue_atoms, R_pca, center)
    active_ca_coords = (R_pca @ (raw_active_ca - center).T).T + center

    membership_count = _compute_jt_membership(gg, L)

    return {
        "obj":                obj,
        "obj_name":           obj_name,
        "residue_atoms":      residue_atoms,
        "residue_str":        residue_str,
        "active_cif_indices": active_cif_indices,
        "active_cif_set":     active_cif_set,
        "L":                  L,
        "edges":              edges,
        "gg":                 gg,
        "active_ca_coords":   active_ca_coords,
        "membership_count":   membership_count,
    }


# ─── Shared 3-D helpers ───────────────────────────────────────────────────────

def _set_3d_tight_aspect(ax, pts):
    """Tight limits with box_aspect ∝ data extents; extra +X space for vertical scale bar."""
    if len(pts) == 0:
        return np.zeros(3), np.ones(3)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    ranges = np.maximum(mx - mn, 1.0)
    max_r  = float(ranges.max())
    p      = max_r * 0.02
    lo     = (mn - p).copy()
    hi     = (mx + p).copy()
    hi[0] += max_r * 0    # extra right-X space for the vertical scale bar (0.3 before)
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    ax.set_box_aspect(hi - lo)
    return lo, hi


def _clean_3d_ax(ax):
    """Remove all built-in box panes, edges, grid, ticks, and spine lines."""
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor((1, 1, 1, 0))   # fully transparent
    # set_pane_color is the most reliable way to kill face fill in recent mpl
    for set_pc in (ax.xaxis.set_pane_color,
                   ax.yaxis.set_pane_color,
                   ax.zaxis.set_pane_color):
        try:
            set_pc((1, 1, 1, 0))
        except Exception:
            pass
    ax.grid(False)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # Zero-width spines so no axis line is rendered
    for spine in (ax.xaxis.line, ax.yaxis.line, ax.zaxis.line):
        spine.set_linewidth(0)


def _draw_corner_axes(ax, lo, hi, color="#E0E0E0", lw=0.8):
    """Draw 3 axis lines forming the back-left inner corner (azim=-60 view).
    Corner at (lo[0], hi[1], lo[2]) — left (min X), back (max Y), bottom (min Z).
    With azim=-60 the camera is at (-X, +Y), so max-Y = away-from-camera = 'back'
    and min-X = camera-left = 'left'.  These three lines are the only ones shown."""
    xn, yn, zn = float(lo[0]), float(lo[1]), float(lo[2])
    xx, yx, zx = float(hi[0]), float(hi[1]), float(hi[2])
    # Corner at (xn, yx, zn): back-left-bottom
    ax.plot([xn, xx], [yx, yx], [zn, zn], color=color, lw=lw, zorder=0)  # X along back-bottom
    ax.plot([xn, xn], [yn, yx], [zn, zn], color=color, lw=lw, zorder=0)  # Y along left-bottom
    ax.plot([xn, xn], [yx, yx], [zn, zx], color=color, lw=lw, zorder=0)  # Z along back-left up


def _style_legend(leg, facecolor="#FAF7F0"):
    frame = leg.get_frame()
    frame.set_facecolor(facecolor)
    frame.set_alpha(1.0)
    frame.set_edgecolor("none")


def _add_scale_bars_3d(ax, lo, hi, data_max_x, z_bot, z_top,
                       vcol="#999999", bar_len=4.5):
    """Draw both scale bars on the right-back of the plot.

    Vertical bar (gray): full protein Z (PC1) height, right side, back face.
    Horizontal bar (black, 4.5 Å): at the BASE of the vertical bar, same
    x-position and y-position, so both share one anchor point.
    """
    yback     = float(hi[1])                               # back face
    right_pad = float(hi[0]) - float(data_max_x)
    x_pos     = float(data_max_x) + right_pad * 0.45      # into right padding
    z_bot     = float(z_bot)
    z_top     = float(z_top)
    z_range   = float(hi[2]) - float(lo[2])
    full_len  = z_top - z_bot
    vtick     = 2.0            # Å, end ticks on vertical bar
    htick     = z_range * 0.025   # end ticks on horizontal bar

    # ── Vertical scale bar ────────────────────────────────────────────────
    ax.plot([x_pos, x_pos], [yback, yback], [z_bot, z_top],
            color=vcol, lw=1.5, zorder=0)
    for zp in (z_bot, z_top):
        ax.plot([x_pos - vtick, x_pos + vtick], [yback, yback], [zp, zp],
                color=vcol, lw=1.5, zorder=0)
    scale_str = f"{full_len:.1f}"
    scale_str_len = len(scale_str) - 1 # sub 1 for point, only numerals
    ax.text(x_pos + scale_str_len * 0.5, yback, z_top - (0.05 * z_range),
            f"{scale_str} Å", color=vcol, fontsize=FS_SCALE_BAR,
            ha="left", va="center", zorder=1)

    # ── Horizontal 4.5-Å bar at the base of the vertical bar ─────────────
    x_left  = x_pos - bar_len / 2
    x_right = x_pos + bar_len / 2
    ax.plot([x_left, x_right], [yback, yback], [z_bot, z_bot],
            color="black", linewidth=0.8, zorder=10)
    for xp in (x_left, x_right):
        ax.plot([xp, xp], [yback, yback], [z_bot - htick, z_bot + htick],
                color="black", linewidth=0.8, zorder=10)
    ax.text((x_left + x_right) / 2, yback, z_bot - htick * 2.5,
            f"{bar_len} Å", ha="center", va="top", fontsize=FS_SCALE_BAR,
            color="black", zorder=10)


# ─── Panel 1: full 3-D structure ──────────────────────────────────────────────

def plot_panel1(ax, data, annotate=True):
    """Backbone chain + all CA dots (black) + designable CA dots (red)."""
    ra      = data["residue_atoms"]
    act_set = data["active_cif_set"]

    ca_seq = [res["ca"] for res in ra if res["ca"] is not None]
    ca_arr = np.array(ca_seq)

    # Backbone chain
    ax.plot(ca_arr[:, 0], ca_arr[:, 1], ca_arr[:, 2],
            color=COL_BB, lw=2.0, alpha=0.55, zorder=1)

    # All α-carbon atoms (black)
    ax.scatter(ca_arr[:, 0], ca_arr[:, 1], ca_arr[:, 2],
               c="black", s=1, alpha=1.0, zorder=2, linewidths=0)

    # Designable position CAs (red, slightly transparent overlay)
    des_ca = np.array([
        res["ca"] for cif_idx, res in enumerate(ra)
        if cif_idx in act_set and res["ca"] is not None
    ])
    if len(des_ca):
        ax.scatter(des_ca[:, 0], des_ca[:, 1], des_ca[:, 2],
                   c=COL_NON_OPT, s=12, alpha=0.30, zorder=3, linewidths=0)

    lo, hi = _set_3d_tight_aspect(ax, ca_arr)
    _clean_3d_ax(ax)
    _draw_corner_axes(ax, lo, hi)
    _add_scale_bars_3d(ax, lo, hi,
        data_max_x=float(ca_arr[:, 0].max()),
        z_bot=float(ca_arr[:, 2].min()),
        z_top=float(ca_arr[:, 2].max()))

    ax.view_init(VIEW_ELEV, VIEW_AZIM)

    if annotate:
        ax.set_title(
            f"Designable positions in context\nof full 3D structure"
            f"\n(L={len(ra)}, L_opt={data['L']})",
            fontsize=FS_TITLE, pad=2,
        )
        legend_elements = [
            Line2D([0], [0], color=COL_BB, lw=2.0, label="Backbone"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="black", markersize=3, lw=0,
                   label="\u03b1-carbon atom"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COL_NON_OPT, markersize=5, lw=0,
                   label="Designable position"),
        ]
        leg = ax.legend(handles=legend_elements, fontsize=FS_LEGEND,
                        loc="upper left", frameon=True, handlelength=1.5)
        _style_legend(leg)


# ─── Panel 2: active residues + ball-and-stick + contact edges ───────────────

def plot_panel2(ax, data, annotate=True):
    """Backbone lines + ball-and-stick (red) + contact edges (black) + CA dots."""
    ra        = data["residue_atoms"]
    act_inds  = data["active_cif_indices"]
    active_ca = data["active_ca_coords"]
    edges     = data["edges"]
    obj       = data["obj"]

    # Backbone chain through active CAs (connectivity line)
    if len(active_ca) > 1:
        ax.plot(active_ca[:, 0], active_ca[:, 1], active_ca[:, 2],
                color=COL_BB, lw=2.5, alpha=0.55, zorder=1)

    # Contact edges (dark blue)
    for i, j in edges:
        ax.plot(
            [active_ca[i, 0], active_ca[j, 0]],
            [active_ca[i, 1], active_ca[j, 1]],
            [active_ca[i, 2], active_ca[j, 2]],
            color=COL_EDGE_CONTACT, lw=0.5, alpha=0.65, zorder=2,
        )

    # Ball-and-stick for each active residue
    for local_idx, cif_idx in enumerate(act_inds):
        res      = ra[cif_idx]
        aa_char  = obj.WT_seq[local_idx] if local_idx < len(obj.WT_seq) else "G"
        bonds    = AA_BONDS.get(aa_char, [("N","CA"),("CA","C"),("C","O")])
        ad       = res["atom_dict"]

        # Sticks
        for (a1, a2) in bonds:
            if a1 in ad and a2 in ad:
                p1, p2 = ad[a1], ad[a2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=COL_NON_OPT, lw=0.8, alpha=0.28, zorder=3)

        # Balls (all heavy atoms in this residue)
        all_coords = np.array(list(ad.values()))
        if len(all_coords):
            ax.scatter(all_coords[:, 0], all_coords[:, 1], all_coords[:, 2],
                       c=COL_NON_OPT, s=4, alpha=0.28, zorder=4, linewidths=0)

    # α-carbon dots (black, on top)
    ax.scatter(active_ca[:, 0], active_ca[:, 1], active_ca[:, 2],
               c="black", s=4, alpha=1.0, zorder=5, linewidths=0)

    lo, hi = _set_3d_tight_aspect(ax, active_ca)
    _clean_3d_ax(ax)
    _draw_corner_axes(ax, lo, hi)
    _add_scale_bars_3d(ax, lo, hi,
        data_max_x=float(active_ca[:, 0].max()),
        z_bot=float(active_ca[:, 2].min()),
        z_top=float(active_ca[:, 2].max()))

    ax.view_init(VIEW_ELEV, VIEW_AZIM)

    if annotate:
        ax.set_title(
            f"Contact graph\n(|N|={data['L']}, |E|={len(edges)})",
            fontsize=FS_TITLE, pad=2,
        )
        legend_elements = [
            Line2D([0], [0], color=COL_BB, lw=2.5, label="Backbone"),
            Line2D([0], [0], color=COL_EDGE_CONTACT, lw=0.5, alpha=0.8,
                   label="Contact edge"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COL_NON_OPT, markersize=4, lw=0, label="Atom"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="black", markersize=4, lw=0,
                   label="\u03b1-carbon atom"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COL_NON_OPT, markersize=6, lw=0,
                   label="Designable position"),
        ]
        leg = ax.legend(handles=legend_elements, fontsize=FS_LEGEND,
                        loc="upper left", frameon=True, handlelength=1.5)
        _style_legend(leg)


# ─── Panel 3: JT membership colouring ────────────────────────────────────────

def plot_panel3(ax, fig, data, annotate=True):
    """CA positions coloured by # JT nodes the residue appears in (global Greens)."""
    active_ca = data["active_ca_coords"]
    edges     = data["edges"]
    mc        = data["membership_count"]

    norm   = Normalize(vmin=1, vmax=GLOBAL_MAX_MEMBERSHIP)
    cmap   = cm.Greens
    max_mc = int(mc.max())

    for i, j in edges:
        ax.plot(
            [active_ca[i, 0], active_ca[j, 0]],
            [active_ca[i, 1], active_ca[j, 1]],
            [active_ca[i, 2], active_ca[j, 2]],
            color=COL_EDGE_CONTACT, lw=0.5, alpha=0.65, zorder=1,
        )

    ax.scatter(
        active_ca[:, 0], active_ca[:, 1], active_ca[:, 2],
        c=mc.tolist(), cmap=cmap, norm=norm,
        s=25, alpha=0.90, zorder=3,
        edgecolors="black", linewidths=0.5,
    )

    lo, hi = _set_3d_tight_aspect(ax, active_ca)
    _clean_3d_ax(ax)
    _draw_corner_axes(ax, lo, hi)
    _add_scale_bars_3d(ax, lo, hi,
        data_max_x=float(active_ca[:, 0].max()),
        z_bot=float(active_ca[:, 2].min()),
        z_top=float(active_ca[:, 2].max()))
    ax.view_init(VIEW_ELEV, VIEW_AZIM)

    if annotate:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.08,
                          label="# JT nodes position is contained in",
                          orientation="vertical")
        cb.ax.axhline(y=max_mc, color="black", linewidth=1.2, linestyle="-")

        ax.set_title(
            f"Membership of positions\nin junction tree nodes\n(max={max_mc})",
            fontsize=FS_TITLE, pad=2,
        )
        edge_proxy = Line2D([0], [0], color=COL_EDGE_CONTACT, lw=0.5, label="Contact edge")
        node_proxy = Line2D([0], [0], marker="o", color="w",
                            markerfacecolor="white", markeredgecolor="black",
                            markeredgewidth=0.5, markersize=5, lw=0,
                            label="Position (node)")
        leg = ax.legend(handles=[edge_proxy, node_proxy], fontsize=FS_LEGEND,
                        loc="upper left", frameon=True, handlelength=1.5)
        _style_legend(leg)


# ─── Panel 4: 2-D PCA + convex-hull overlays (Blues by clique size) ──────────

def _expand_hull_verts_2d(pts, buffer):
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None
    verts    = pts[hull.vertices]
    centroid = verts.mean(axis=0)
    dirs     = verts - centroid
    norms    = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms    = np.where(norms < 1e-8, 1.0, norms)
    expanded = verts + (dirs / norms) * buffer
    return np.vstack([expanded, expanded[0]])


def plot_panel4(ax, fig, data, annotate=True):
    """2-D PCA + JT-node hulls coloured by clique size (Blues, global scale)."""
    active_ca = data["active_ca_coords"]
    gg        = data["gg"]
    L         = data["L"]

    sz_norm = Normalize(vmin=1, vmax=GLOBAL_MAX_CLIQUE_SIZE)
    sz_cmap = cm.Blues

    if L < 2:
        ax.text(0.5, 0.5, "Too few residues", ha="center", va="center",
                transform=ax.transAxes)
        if annotate:
            ax.set_title("JT node hulls (2D PCA)", fontsize=FS_TITLE)
        return

    pca = PCA(n_components=min(2, L))
    coords_2d = pca.fit_transform(active_ca)
    if coords_2d.shape[1] == 1:
        coords_2d = np.hstack([coords_2d, np.zeros((L, 1))])
    # Swap so PC1 (most variance) is on the Y axis — uses tall plot space better
    coords_2d = coords_2d[:, [1, 0]]

    buf = HULL_BUFFER * 0.7   # 2.1 Å — generous enough that boundary clears node dots

    for sg_idx, sg in enumerate(gg.subgraphs):
        if not isinstance(sg, JunctionTree):
            continue
        g_inds = gg.subgraph_node_indices[sg_idx]

        for c in range(sg.n_nodes):
            members     = [g_inds[li] for li in sg.index_to_nodes[c]]
            pts         = coords_2d[members]
            clique_size = len(sg.index_to_nodes[c])
            color       = sz_cmap(sz_norm(clique_size))
            is_root       = (c == sg.root)
            outline_lw    = 1.8 if is_root else 0.8
            outline_color = "black" if is_root else (0, 0, 0, 0.4)
            hull_zorder   = 4 if is_root else 3   # root always in front
            k             = len(pts)

            # Encode fill alpha directly in RGBA so edge stays at alpha=1
            face_rgba = (*color[:3], 0.22)

            if k == 1:
                ax.add_patch(plt.Circle(pts[0], buf,
                                        facecolor=face_rgba,
                                        edgecolor=outline_color,
                                        linewidth=outline_lw, zorder=hull_zorder))
            elif k == 2:
                d  = pts[1] - pts[0]
                dn = np.linalg.norm(d)
                if dn > 1e-8:
                    d_hat = d / dn
                    perp  = np.array([-d[1], d[0]]) / dn * buf
                    corners = np.array([
                        pts[0] - d_hat * buf + perp,
                        pts[1] + d_hat * buf + perp,
                        pts[1] + d_hat * buf - perp,
                        pts[0] - d_hat * buf - perp,
                    ])
                    ax.add_patch(mpatches.Polygon(corners, closed=True,
                                                  facecolor=face_rgba,
                                                  edgecolor=outline_color,
                                                  linewidth=outline_lw, zorder=hull_zorder))
                else:
                    ax.add_patch(plt.Circle(pts[0], buf,
                                            facecolor=face_rgba,
                                            edgecolor=outline_color,
                                            linewidth=outline_lw, zorder=hull_zorder))
            else:
                poly_pts = _expand_hull_verts_2d(pts, buf)
                if poly_pts is not None:
                    ax.add_patch(
                        mpatches.Polygon(poly_pts[:-1], closed=True,
                                         facecolor=face_rgba,
                                         edgecolor=outline_color,
                                         linewidth=outline_lw, zorder=hull_zorder)
                    )

    # Design positions — black dots on top
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
               c="black", s=18, zorder=5, alpha=1.0, linewidths=0)

    ax.set_aspect("equal", adjustable="datalim")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC2 ({var[1]:.0%})" if len(var) > 1 else "PC2", fontsize=FS_LEGEND)
    ax.set_ylabel(f"PC1 ({var[0]:.0%})", fontsize=FS_LEGEND)
    ax.tick_params(labelbottom=False, labelleft=False)

    if annotate:
        sm4 = ScalarMappable(cmap=sz_cmap, norm=sz_norm)
        sm4.set_array([])
        fig.colorbar(sm4, ax=ax, shrink=0.6, pad=0.03,
                     label="# positions in JT node", orientation="vertical")

        ax.set_title(
            "Membership of positions in\njunction tree nodes as hulls (2D PCA)",
            fontsize=FS_TITLE,
        )
        dot_proxy  = Line2D([0], [0], marker="o", color="w",
                            markerfacecolor="black", markersize=5, lw=0,
                            label="Design position")
        hull_proxy = mpatches.Patch(facecolor=sz_cmap(sz_norm(5)),
                                    edgecolor="black", linewidth=0.8,
                                    alpha=0.35, label="JT node (hull)")
        leg = ax.legend(handles=[dot_proxy, hull_proxy], fontsize=FS_LEGEND,
                        loc="upper left", frameon=True)
        _style_legend(leg)


# ─── Panel 5: JT network ──────────────────────────────────────────────────────

def _spring_layout_same_y(pos, G, y_tol_frac=0.02, min_sep_frac=0.05,
                           k_spring=0.30, iterations=300):
    """
    For each horizontal layer of nodes (same y in dot layout), apply 1-D spring
    relaxation in x so that nodes spread apart while each stays near its
    parent's x-coordinate (mean of predecessors).

    Forces:
      - Soft-core repulsion between all node pairs at the same y-level.
      - Spring attraction toward each node's parent anchor.
    Post-pass enforces minimum separation symmetrically.
    """
    pos   = dict(pos)
    nodes = list(pos.keys())
    if len(nodes) < 2:
        return pos

    all_y   = [pos[n][1] for n in nodes]
    all_x   = [pos[n][0] for n in nodes]
    y_range = max(all_y) - min(all_y) if len(set(all_y)) > 1 else 1.0
    x_range = max(all_x) - min(all_x) if len(set(all_x)) > 1 else 1.0
    y_tol   = max(y_range * y_tol_frac, 1e-3)
    min_sep = max(x_range * min_sep_frac, 5.0)

    # Group into horizontal layers by y-coordinate bucket
    y_groups: dict = defaultdict(list)
    for n in nodes:
        bucket = round(pos[n][1] / y_tol)
        y_groups[bucket].append(n)

    for group in y_groups.values():
        if len(group) < 2:
            continue

        # Anchor for each node = mean x-coord of its graph predecessors
        anchors = {}
        for n in group:
            preds = list(G.predecessors(n))
            anchors[n] = (float(np.mean([pos[p][0] for p in preds]))
                          if preds else float(pos[n][0]))

        x    = {n: float(pos[n][0]) for n in group}
        step = min_sep * 0.05

        for _ in range(iterations):
            forces = {n: 0.0 for n in group}
            ns = list(group)

            # Soft-core repulsion (constant beyond min_sep/4, Coulomb-like inside)
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    ni, nj = ns[i], ns[j]
                    dx  = x[nj] - x[ni]
                    sgn = 1.0 if dx >= 0 else -1.0
                    d   = max(abs(dx), min_sep * 0.25)
                    f   = (min_sep / d) ** 2
                    forces[ni] -= f * sgn
                    forces[nj] += f * sgn

            # Spring attraction toward parent anchor
            for n in group:
                forces[n] += k_spring * (anchors[n] - x[n]) / min_sep

            for n in group:
                x[n] += step * forces[n]

        # Symmetric post-pass: enforce minimum separation
        sorted_nodes = sorted(group, key=lambda n: x[n])
        for _ in range(30):
            moved = False
            for i in range(1, len(sorted_nodes)):
                a, b  = sorted_nodes[i - 1], sorted_nodes[i]
                gap   = x[b] - x[a]
                if gap < min_sep:
                    push  = (min_sep - gap) / 2.0
                    x[a] -= push
                    x[b] += push
                    moved = True
            if not moved:
                break

        for n in group:
            pos[n] = (x[n], pos[n][1])

    return pos


def plot_panel5(ax, fig, data, annotate=True):
    """JT network: uniform-size nodes coloured by clique size (Blues)."""
    gg = data["gg"]

    G          = nx.DiGraph()
    node_sz    = {}
    root_nodes = set()
    NODE_PX    = 180  # uniform node size in px²

    nid = 0
    for sg_idx, sg in enumerate(gg.subgraphs):
        g_inds = gg.subgraph_node_indices[sg_idx]

        if isinstance(sg, JunctionTree):
            c2n = {}
            for c in range(sg.n_nodes):
                c2n[c] = nid
                node_sz[nid] = len(sg.index_to_nodes[c])
                G.add_node(nid)
                nid += 1
            for u, v in sg.graph.edges():
                G.add_edge(c2n[u], c2n[v])
            root_nodes.add(c2n[sg.root])

        elif isinstance(sg, (Tree, DisconnectedGraph)):
            v2n = {}
            for li in range(sg.n_nodes):
                v2n[li] = nid
                node_sz[nid] = 1
                G.add_node(nid)
                nid += 1
            if isinstance(sg, Tree):
                for u, v in sg.graph.edges():
                    G.add_edge(v2n[u], v2n[v])
                root_nodes.add(v2n[sg.root])

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty JT", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return

    ordered = list(G.nodes())

    sz_norm   = Normalize(vmin=1, vmax=GLOBAL_MAX_CLIQUE_SIZE)
    sz_cmap   = cm.Blues
    # Embed fill alpha=0.6 in RGBA so edgecolors remain at alpha=1
    node_rgba = {n: (*sz_cmap(sz_norm(node_sz[n]))[:3], 0.6) for n in ordered}

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Granksep=1.0")
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=3.0 / max(len(ordered) ** 0.5, 1))

    pos = _spring_layout_same_y(pos, G)

    # Scale y-positions so the tree fills the panel height.
    # Target: y_extent ≈ x_extent × PANEL_ASPECT (h/w ≈ 1.4 for this panel).
    _PANEL_ASPECT = 1.4
    _xs = [pos[n][0] for n in pos]
    _ys = [pos[n][1] for n in pos]
    x_ext = max(max(_xs) - min(_xs), 1.0)
    y_ext = max(max(_ys) - min(_ys), 1.0)
    target_y = x_ext * _PANEL_ASPECT
    if y_ext < target_y:
        y_mid = (max(_ys) + min(_ys)) / 2
        y_s   = target_y / y_ext
        pos   = {n: (x, y_mid + (y - y_mid) * y_s) for n, (x, y) in pos.items()}

    sizes_list = [NODE_PX] * len(ordered)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        nodelist=ordered,
        node_size=sizes_list,
        edge_color="#222222", width=0.9,
        arrows=True, arrowsize=8,
        alpha=1.0,
        connectionstyle="arc3,rad=0.0",
    )

    non_root = [n for n in ordered if n not in root_nodes]
    if non_root:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=non_root,
            node_size=[NODE_PX] * len(non_root),
            node_color=[node_rgba[n] for n in non_root],
            linewidths=0.8, edgecolors="black",
        )

    root_list = list(root_nodes)
    if root_list:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=root_list,
            node_size=[NODE_PX] * len(root_list),
            node_color=[node_rgba[n] for n in root_list],
            linewidths=2.5, edgecolors="black",
        )

    K = G.number_of_nodes()
    E = G.number_of_edges()
    ax.set_axis_off()
    ax.set_aspect('auto')   # allow tree to fill full panel height

    # Expand axes limits so nodes (with visual radius) stay inside and don't
    # bleed into adjacent panels 4 or 6.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xp = (xmax - xmin) * 0.15
    yp = (ymax - ymin) * 0.10
    ax.set_xlim(xmin - xp, xmax + xp * 1.8)   # extra right margin for panel-6 labels
    ax.set_ylim(ymin - yp, ymax + yp)

    if annotate:
        sm5 = ScalarMappable(cmap=sz_cmap, norm=sz_norm)
        sm5.set_array([])
        fig.colorbar(sm5, ax=ax, shrink=0.6, pad=0.03,
                     label="# positions in JT node", orientation="vertical")
        ax.set_title(f"Junction tree topology\n(|N|={K}, |E|={E}; outline=root)",
                     fontsize=FS_TITLE)


# ─── Panel 6: two stacked cardinality plots ───────────────────────────────────

def plot_panel6(ax_top, ax_bot, data, ymax, annotate=True):
    """Top: parent cardinality; Bottom: node cardinality. Shared x, global y."""
    gg = data["gg"]

    node_sizes, parent_sizes = [], []
    for sg in gg.subgraphs:
        if isinstance(sg, JunctionTree):
            node_sizes.extend(sg.clique_sizes)
            parent_sizes.extend(sg.clique_parent_sizes)

    BAR_COLOR = "#888888"

    if not node_sizes:
        for ax in (ax_top, ax_bot):
            ax.text(0.5, 0.5, "No JT cliques", ha="center", va="center",
                    transform=ax.transAxes)
        ax_top.set_ylabel("Node cardinality", fontsize=FS_LEGEND)
        ax_bot.set_ylabel("Edge cardinality", fontsize=FS_LEGEND)
        if annotate:
            ax_top.set_title("JT node & edge cardinalities", fontsize=FS_TITLE)
        return

    ns  = np.array(node_sizes)
    ps  = np.array(parent_sizes)
    idx = np.arange(len(ns))

    ax_top.bar(idx, ns, color=BAR_COLOR)
    ax_bot.bar(idx, ps, color=BAR_COLOR)

    for ax in (ax_top, ax_bot):
        ax.set_ylim(0, ymax * 1.05)
        ax.tick_params(labelsize=FS_LEGEND)
        # Force x-axis to have only integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax_top.tick_params(labelbottom=False)
    ax_bot.set_xlabel("Node index", fontsize=FS_LEGEND)
    ax_top.set_ylabel("Node cardinality", fontsize=FS_LEGEND)
    ax_bot.set_ylabel("Edge cardinality", fontsize=FS_LEGEND)

    if annotate:
        ax_top.set_title(
            f"JT node & edge cardinalities\n(|N|={len(ns)}, "
            f"mean node cardinality={ns.mean():.1f})",
            fontsize=FS_TITLE,
        )


# ─── Figure builder ───────────────────────────────────────────────────────────

def make_protein_figure(obj_name, structure_path, outdir, panel6_ymax, clean_fig_width):
    """Build + save clean and annotated figures for one protein.

    Returns the clean figure width (inches) after gap removal so the
    titles/legends/colorbars strip can be made the same width.
    """
    print(f"\n{'─'*60}")
    print(f"  {obj_name}: loading data …")
    data = load_protein_data(obj_name, structure_path)
    print(f"  {obj_name}: L={data['L']}, |E|={len(data['edges'])}, "
          f"N_CIF={len(data['residue_atoms'])}")

    # clean_fig_width = 26.0  # fallback; overwritten by annotate=False pass

    for annotate in (False, True):
        fig = plt.figure(figsize=(32, 6))
        # Clean version: give more width to the 3-D panels (no legends/colorbars)
        # Annotated version: equal widths to accommodate colorbar space (need not be equal, should tune by hand)
        w   = [1.2, 1.2, 1.2, 1.0, 1.0, 1.1] if not annotate else [1, 1, 1, 1, 1, 1]
        wsp = 0.06 if not annotate else 0.35
        gs  = GridSpec(2, 6, figure=fig, hspace=0.0, wspace=wsp, # only one row, so no height needed. 
                       width_ratios=w)

        ax1 = fig.add_subplot(gs[:, 0], projection="3d")
        ax2 = fig.add_subplot(gs[:, 1], projection="3d")
        ax3 = fig.add_subplot(gs[:, 2], projection="3d")
        ax4 = fig.add_subplot(gs[:, 3])
        ax5 = fig.add_subplot(gs[:, 4])
        ax6_top = fig.add_subplot(gs[0, 5])
        ax6_bot = fig.add_subplot(gs[1, 5], sharex=ax6_top)

        print(f"  {obj_name}: drawing panels (annotate={annotate}) …")
        plot_panel1(ax1, data, annotate=annotate)
        plot_panel2(ax2, data, annotate=annotate)
        plot_panel3(ax3, fig, data, annotate=annotate)
        plot_panel4(ax4, fig, data, annotate=annotate)
        plot_panel5(ax5, fig, data, annotate=annotate)
        plot_panel6(ax6_top, ax6_bot, data, panel6_ymax, annotate=annotate)

        # Align 3-D panels vertically with the 2-D panels
        plt.tight_layout(pad=0.5 if not annotate else 1.2)
        pos_ref = ax4.get_position()
        for ax_3d in (ax1, ax2, ax3):
            pos = ax_3d.get_position()
            ax_3d.set_position([pos.x0, pos_ref.y0, pos.width, pos_ref.height])

        # ── Remove gaps between 3-D panels and shrink figure width ────────────
        fig_w, fig_h = fig.get_size_inches()

        def _ax_inches(ax_):
            p = ax_.get_position()
            return p.x0 * fig_w, p.y0 * fig_h, p.width * fig_w, p.height * fig_h

        x1i, y1i, w1i, h1i   = _ax_inches(ax1)
        x2i, y2i, w2i, h2i   = _ax_inches(ax2)
        x3i, y3i, w3i, h3i   = _ax_inches(ax3)
        x4i, y4i, w4i, h4i   = _ax_inches(ax4)
        x5i, y5i, w5i, h5i   = _ax_inches(ax5)
        x6ti, y6ti, w6ti, h6ti = _ax_inches(ax6_top)
        x6bi, y6bi, w6bi, h6bi = _ax_inches(ax6_bot)

        gap12     = x2i - (x1i + w1i)
        gap23     = x3i - (x2i + w2i)
        total_gap = gap12 + gap23

        nx2  = x1i + w1i
        nx3  = nx2  + w2i
        nx4  = x4i  - total_gap
        nx5  = x5i  - total_gap
        nx6t = x6ti - total_gap
        nx6b = x6bi - total_gap

        new_W = fig_w - total_gap
        fig.set_size_inches(new_W, fig_h)

        for ax_, xi, yi, wi, hi_ in [
            (ax1,     x1i,  y1i,  w1i,  h1i),
            (ax2,     nx2,  y2i,  w2i,  h2i),
            (ax3,     nx3,  y3i,  w3i,  h3i),
            (ax4,     nx4,  y4i,  w4i,  h4i),
            (ax5,     nx5,  y5i,  w5i,  h5i),
            (ax6_top, nx6t, y6ti, w6ti, h6ti),
            (ax6_bot, nx6b, y6bi, w6bi, h6bi),
        ]:
            ax_.set_position([xi / new_W, yi / fig_h, wi / new_W, hi_ / fig_h])

        if not annotate:
            clean_fig_width = new_W

        suffix = "_titles_legends_cbars" if annotate else ""
        out_path = os.path.join(outdir, f"{obj_name}{suffix}.png")
        plt.savefig(out_path, dpi=400, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out_path}")

    return clean_fig_width


# ─── Standalone titles / legends / colorbars strip ────────────────────────────

def make_titles_legends_cbars_strip(outdir, figwidth=32.0):
    """
    Create a figure of the same total width as the protein figures, with 6
    columns (one per panel).  Each column stacks: title / legend / colorbar.
    Intended to be placed above the protein rows in Overleaf.
    """
    fig = plt.figure(figsize=(figwidth, 3.0))
    gs  = GridSpec(3, 6, figure=fig,
                   height_ratios=[1.0, 1.0, 0.6],
                   hspace=0.05, wspace=0.35)

    # ── titles ────────────────────────────────────────────────────────────────
    titles = [
        "Designable positions in\ncontext of full 3D structure",
        "Contact graph\n(designable positions)",
        "Membership of positions\nin junction tree nodes",
        "JT node hulls\n(2D PCA)",
        "Junction tree topology\n(outline = root)",
        "JT node & edge\ncardinalities",
    ]
    for col, title in enumerate(titles):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, title, ha="center", va="center",
                fontsize=FS_TITLE, transform=ax.transAxes, fontweight="bold")
        ax.axis("off")

    # ── legends ───────────────────────────────────────────────────────────────
    panel_legends = [
        # Panel 1
        [Line2D([0],[0], color=COL_BB, lw=2.0, label="Backbone"),
         Line2D([0],[0], marker="o", color="w", markerfacecolor="black",
                markersize=3, lw=0, label="\u03b1-carbon atom"),
         Line2D([0],[0], marker="o", color="w", markerfacecolor=COL_NON_OPT,
                markersize=5, lw=0, label="Designable position")],
        # Panel 2
        [Line2D([0],[0], color=COL_BB, lw=2.5, label="Backbone"),
         Line2D([0],[0], color=COL_EDGE_CONTACT, lw=0.5, alpha=0.8,
                label="Contact edge"),
         Line2D([0],[0], marker="o", color="w", markerfacecolor=COL_NON_OPT,
                markersize=4, lw=0, label="Atom"),
         Line2D([0],[0], marker="o", color="w", markerfacecolor="black",
                markersize=4, lw=0, label="\u03b1-carbon atom")],
        # Panel 3
        [Line2D([0],[0], color=COL_EDGE_CONTACT, lw=0.5, label="Contact edge"),
         Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=0.5,
                markersize=5, lw=0, label="Position (node)")],
        # Panel 4
        [Line2D([0],[0], marker="o", color="w", markerfacecolor="black",
                markersize=5, lw=0, label="Design position"),
         mpatches.Patch(facecolor=cm.Blues(0.5), edgecolor="black",
                        linewidth=0.8, alpha=0.35, label="JT node (hull)")],
        # Panel 5
        [mpatches.Patch(facecolor=cm.Blues(0.5), edgecolor="black",
                        linewidth=0.8, alpha=0.6, label="JT node"),
         mpatches.Patch(facecolor=cm.Blues(0.5), edgecolor="black",
                        linewidth=2.5, alpha=0.6, label="Root node")],
        # Panel 6 – no legend
        [],
    ]

    for col, handles in enumerate(panel_legends):
        # Panels 1 & 2 have no colorbar: span rows 1+2 so the legend sits
        # centred in the larger space below the title (avoids title overlap).
        ax = fig.add_subplot(gs[1:3, col] if col in (0, 1) else gs[1, col])
        ax.axis("off")
        if handles:
            # Panel 2 (col 1) legend shifted slightly below centre
            kw = dict(fontsize=FS_LEGEND, frameon=True, ncol=1, handlelength=1.5)
            if col == 1:
                leg = ax.legend(handles=handles, loc="center",
                                bbox_to_anchor=(0.5, 0.40),
                                bbox_transform=ax.transAxes, **kw)
            else:
                leg = ax.legend(handles=handles, loc="center", **kw)
            _style_legend(leg)

    # ── colorbars ─────────────────────────────────────────────────────────────
    # Panel 1, 2, 6: no colorbar
    cbar_specs = {
        2: (cm.Greens, Normalize(1, GLOBAL_MAX_MEMBERSHIP),
            "# JT nodes containing position"),
        3: (cm.Blues, Normalize(1, GLOBAL_MAX_CLIQUE_SIZE),
            "# positions in JT node"),
        4: (cm.Blues, Normalize(1, GLOBAL_MAX_CLIQUE_SIZE),
            "# positions in JT node"),
    }
    for col in range(6):
        if col in (0, 1):
            continue  # legend already spans rows 1:3 for these columns
        ax = fig.add_subplot(gs[2, col])
        ax.axis("off")
        if col in cbar_specs:
            cmap, norm, label = cbar_specs[col]
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                              fraction=0.6, pad=-0.20, aspect=20)
            cb.set_label(label, fontsize=FS_LEGEND)
            cb.ax.tick_params(labelsize=FS_LEGEND - 1)

    out_path = os.path.join(outdir, "titles_legends_cbars.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved strip → {out_path}")


# ─── First-pass: compute global panel-6 y-axis max ────────────────────────────

def compute_global_panel6_ymax(structure_dir):
    """Compute max parent cardinality across all proteins (for shared y-axis)."""
    ymax = 1
    for obj_name, cif_fname in STRUCTURE_PATHS_DICT.items():
        structure_path = os.path.join(structure_dir, cif_fname)
        if not os.path.exists(structure_path):
            continue
        try:
            obj = OracleObjective(obj_name=obj_name)
            edges, _ = residue_contact_map(
                structure_path, obj.WT_seq,
                edgelist=True, verbose=False,
                active_inds=obj.active_inds,
                threshold=THRESHOLD,
            )
            L  = len(obj.D)
            gg = GeneralizedGraph(L, edges, verbose=False)
            for sg in gg.subgraphs:
                if isinstance(sg, JunctionTree) and sg.clique_parent_sizes:
                    ymax = max(ymax, max(sg.clique_parent_sizes))
        except Exception:
            pass
    return ymax


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    outdir        = "experiments/proteins/vis_graphs"
    structure_dir = os.path.join(os.environ["DIR_DATA"], "af3_structures")
    print(f"WARNING: Script not yet edited for \"annotated\" plots--legends, titles, colorbars, and plot spacings may be off in xxx_titles_legends_cbars.png plots figures.")

    print("Computing global panel-6 y-axis range …")
    panel6_ymax = compute_global_panel6_ymax(structure_dir)
    print(f"  GLOBAL_MAX_PANEL6_Y = {panel6_ymax}")

    os.makedirs(outdir, exist_ok=True)

    clean_fig_width = 26.0  # fallback; overwritten by first successful protein
    for obj_name, cif_fname in STRUCTURE_PATHS_DICT.items():
        structure_path = os.path.join(structure_dir, cif_fname)
        if not os.path.exists(structure_path):
            print(f"WARNING: CIF not found for {obj_name}: {structure_path}")
            continue
        try:
            w = make_protein_figure(obj_name, structure_path, outdir, panel6_ymax, clean_fig_width)
            if w is not None:
                clean_fig_width = w   # all proteins give the same gap/width
        except Exception as e:
            print(f"ERROR processing {obj_name}: {e}")
            traceback.print_exc()

    make_titles_legends_cbars_strip(outdir, figwidth=clean_fig_width)
    print("\nDone.")


if __name__ == "__main__":
    main()
