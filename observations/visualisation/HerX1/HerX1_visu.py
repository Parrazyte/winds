"""
Interactive 3-D twisted-tilted accretion disc (Her X-1 style).

Builds a set of concentric rings that each live in their own tilted
plane. The tilt angle and the "twist" (rotation of the line of nodes)
both vary smoothly from the inner edge of the disc to the outer edge,
which is what produces the nested, precessing-looking warped structure
seen in the classic Her X-1 disc figures. The rings are plotted as a
true 3-D object using matplotlib's mplot3d, so you can click-and-drag
to rotate it and view it "from the side" (or from anywhere else)
interactively, instead of fixing a single 2-D projection in the code.

Default geometry is the average of the BAT and MAXI best-fit values
from Table 1 of Leahy & Frost (2025), "Disk with Corona" model:
    Inner twist        : 75.5 deg   (fixed parameter)
    Inner tilt          : (22.5 + 24.0) / 2  = 23.25 deg
    Outer tilt          : (17   + 24  ) / 2  = 20.5  deg
    Outer twist (added) : (22   + 22  ) / 2  = 22.0  deg
    Inclination          : (85.17 + 85.07) / 2 = 85.12 deg
    Rout / Rin ratio     : 2.1e6 / 5e5 = 4.2  (physical units dropped,
                                                only the ratio matters)

Usage:
    python twisted_tilted_disc.py --cmap viridis
    python twisted_tilted_disc.py --cmap plasma --n_rings 20

Run `python twisted_tilted_disc.py --help` for all options. Once the
window opens, click and drag to rotate the disc freely.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


# ---- Leahy & Frost (2025), Table 1, "Disk with Corona" model --------
# Average of the BAT and MAXI best-fit columns; physical (km) values
# and Rcor,in/Rin are dropped as instructed, only the Rout/Rin ratio
# is kept.
LF25_INNER_TWIST_DEG = 75.5                       # fixed parameter
LF25_INNER_TILT_DEG = (22.5 + 24.0) / 2            # 23.25
LF25_OUTER_TILT_DEG = (17.0 + 24.0) / 2            # 20.5
LF25_OUTER_TWIST_ADD_DEG = (22.0 + 22.0) / 2       # 22.0 (added on top of inner twist)
LF25_INCLINATION_DEG = (85.17 + 85.07) / 2         # 85.12
LF25_R_RATIO = 2.1e6 / 5e5                         # Rout / Rin = 4.2


def rot_x(angle):
    """Rotation matrix about the x-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])


def rot_z(angle):
    """Rotation matrix about the z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])


def build_disc(n_rings, r_min, r_max, tilt_in_deg, tilt_out_deg,
                twist_in_deg, twist_out_deg, n_theta):
    """
    Build the 3-D warped disc.

    tilt(r) ramps linearly from tilt_in_deg (at r_min) to tilt_out_deg
    (at r_max); twist(r) ramps linearly from twist_in_deg to
    twist_out_deg (twist_out_deg is the *total* twist at the outer
    edge, i.e. twist_in_deg + the accumulated twist).

    Returns:
        radii: (n_rings,) ring radii, inner -> outer
        rings: list of (3, n_theta) xyz arrays, inner -> outer
        node_locus: (n_rings, 3) xyz curve through each ring's node
    """
    radii = np.linspace(r_min, r_max, n_rings)
    theta = np.linspace(0, 2 * np.pi, n_theta)

    rings = []
    node_locus = []

    for r in radii:
        frac = (r - r_min) / (r_max - r_min)
        tilt = np.radians(tilt_in_deg + (tilt_out_deg - tilt_in_deg) * frac)
        twist = np.radians(twist_in_deg + (twist_out_deg - twist_in_deg) * frac)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(theta)
        pts = np.vstack([x, y, z])

        pts = rot_x(tilt) @ pts     # tilt the ring out of the disc plane
        pts = rot_z(twist) @ pts    # twist (precess) its line of nodes

        rings.append(pts)

        node_pt = rot_z(twist) @ (rot_x(tilt) @ np.array([r, 0, 0]))
        node_locus.append(node_pt)

    return radii, rings, np.array(node_locus)


def plot_disc(cmap_name="viridis", n_rings=16,
              r_min=1.0, r_max=LF25_R_RATIO,
              tilt_in_deg=LF25_INNER_TILT_DEG, tilt_out_deg=LF25_OUTER_TILT_DEG,
              twist_in_deg=LF25_INNER_TWIST_DEG,
              twist_out_deg=LF25_INNER_TWIST_DEG + LF25_OUTER_TWIST_ADD_DEG,
              inclination_deg=LF25_INCLINATION_DEG, view_azim=0.0,
              n_theta=300, show_node_locus=True, savepath=None,
              visu=False, visu_factor=3.0):

    title = "Twisted\u2013tilted disc (Leahy & Frost 2025 avg. geometry)"

    if visu:
        # Exaggerate the warp so the twisted/tilted shape is easier to
        # read by eye. The tilt angles are scaled up (around 0), and the
        # *accumulated* twist (outer - inner) is scaled up around the
        # inner twist, which just sets the starting orientation and
        # isn't itself part of the warp.
        tilt_in_deg = tilt_in_deg * visu_factor
        tilt_out_deg = tilt_out_deg * visu_factor
        twist_out_deg = twist_in_deg + (twist_out_deg - twist_in_deg) * visu_factor
        title += f"\n(exaggerated {visu_factor:g}\u00d7 for visualization)"

    radii, rings, node_locus = build_disc(
        n_rings, r_min, r_max, tilt_in_deg, tilt_out_deg,
        twist_in_deg, twist_out_deg, n_theta
    )

    cmap = plt.get_cmap(cmap_name).resampled(n_rings)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")

    for i, pts in enumerate(rings):
        ax.plot(pts[0], pts[1], pts[2], color=cmap(i), lw=1.3)

    if show_node_locus:
        ax.plot(node_locus[:, 0], node_locus[:, 1], node_locus[:, 2],
                color="0.3", lw=1.2, ls="--", alpha=0.8, label="Locus of Nodes")
        ax.legend(loc="upper left")

    lim = r_max * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Initial camera orientation: "elevation above the disc plane" is
    # (90 - inclination); azimuth is the compass position around the
    # rim. Both are just the *starting* view -- drag with the mouse to
    # rotate freely from there.
    start_elev = 90.0 - inclination_deg
    ax.view_init(elev=start_elev, azim=view_azim)

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=r_min, vmax=r_max))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("Ring radius: inner \u2192 outer")

    ax.set_title(title)

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200)
    plt.show()
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cmap", default="plasma",
                         help="Any matplotlib colormap name (viridis, plasma, "
                              "cividis, magma, cool, turbo, etc.)")
    parser.add_argument("--n_rings", type=int, default=16)
    parser.add_argument("--r_min", type=float, default=1.0)
    parser.add_argument("--r_max", type=float, default=LF25_R_RATIO,
                         help=f"Default is the Rout/Rin ratio from Leahy & Frost "
                              f"2025, {LF25_R_RATIO:.2f}")
    parser.add_argument("--tilt_in", type=float, default=LF25_INNER_TILT_DEG,
                         help="Inner-edge ring tilt, degrees "
                              "(default: avg. BAT/MAXI inner tilt)")
    parser.add_argument("--tilt_out", type=float, default=LF25_OUTER_TILT_DEG,
                         help="Outer-edge ring tilt, degrees "
                              "(default: avg. BAT/MAXI outer tilt)")
    parser.add_argument("--twist_in", type=float, default=LF25_INNER_TWIST_DEG,
                         help="Inner-edge line-of-nodes twist, degrees "
                              "(default: fixed Leahy & Frost value, 75.5)")
    parser.add_argument("--twist_out", type=float,
                         default=LF25_INNER_TWIST_DEG + LF25_OUTER_TWIST_ADD_DEG,
                         help="Outer-edge (total) line-of-nodes twist, degrees "
                              "(default: inner twist + avg. BAT/MAXI outer twist)")
    parser.add_argument("--inclination", type=float, default=LF25_INCLINATION_DEG,
                         help="Observer inclination, degrees, used only to set "
                              "the STARTING camera elevation (90 - inclination) "
                              "before you rotate interactively "
                              "(default: avg. BAT/MAXI inclination)")
    parser.add_argument("--view_azim", type=float, default=0.0,
                         help="Starting camera azimuth, degrees, before you "
                              "rotate interactively")
    parser.add_argument("--no_node_locus", action="store_true")
    parser.add_argument("--visu", action="store_true",default=True,
                         help="Exaggerate the tilt and twist so the warped "
                              "shape is easier to see (scales the tilt angles "
                              "and the accumulated twist by --visu_factor)")
    parser.add_argument("--visu_factor", type=float, default=2.0,
                         help="Exaggeration factor used when --visu is set "
                              "(default: 3.0)")
    parser.add_argument("--save", default=None,
                         help="Optional path to also save a static snapshot "
                              "of the initial view as a PNG")
    args = parser.parse_args()

    plot_disc(
        cmap_name=args.cmap,
        n_rings=args.n_rings,
        r_min=args.r_min,
        r_max=args.r_max,
        tilt_in_deg=args.tilt_in,
        tilt_out_deg=args.tilt_out,
        twist_in_deg=args.twist_in,
        twist_out_deg=args.twist_out,
        inclination_deg=args.inclination,
        view_azim=args.view_azim,
        show_node_locus=not args.no_node_locus,
        savepath=args.save,
        visu=args.visu,
        visu_factor=args.visu_factor,
    )