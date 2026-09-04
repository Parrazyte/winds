
#unfortunately for these we really don't have time so I'm scraping the plots
#with AI to combine them

def plot_hard():
    import numpy as np
    import matplotlib.pyplot as plt
    # ============================================================
    # Data digitized from the supplied figures
    # ============================================================
    # ------------------------------------------------------------
    # Swift J1727.8-1613
    # Blue diamonds in the top panel of the first figure
    # ------------------------------------------------------------
    E_j1727 = np.array([
        2.5, 3.5, 4.5, 5.5, 6.5, 7.5
    ])
    PD_j1727 = np.array([
        2.55, 3.15, 5.15, 5.28, 4.90, 11.58
    ])
    PDerr_j1727 = np.array([
        0.65, 0.77, 1.00, 1.60, 1.9, 4.15
    ])
    # Approximate energy-bin uncertainties from the horizontal errorbars
    Eerr_j1727 = np.array([
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50
    ])
    # ------------------------------------------------------------
    # Cygnus X-1
    # Black points in the second figure
    # ------------------------------------------------------------
    E_cyg = np.array([
        2.25, 2.75, 3.25, 3.75, 4.25,
        4.75, 5.25, 5.75, 6.25, 6.75, 7.50
    ])
    PD_cyg = np.array([
        3.82, 3.91, 3.75, 4.12, 4.79,
        4.44, 5.81, 6.07, 5.65, 7.50, 5.72
    ])
    PDerr_cyg = np.array([
        0.45, 0.40, 0.40, 0.50, 0.57,
        0.70, 0.80, 0.98, 1.13, 1.50, 1.68
    ])
    # Approximate horizontal uncertainties
    Eerr_cyg = np.array([
        0.40, 0.40, 0.40, 0.40, 0.45,
        0.40, 0.35, 0.40, 0.35, 0.30, 0.50
    ])
    # ------------------------------------------------------------
    # IGR J17091-3624
    # Top panel of the third figure
    # ------------------------------------------------------------
    E_igr = np.array([
        2.5, 3.5, 4.5, 6.5
    ])
    PD_igr = np.array([
        7.25, 8.25, 9.00, 13.7
    ])
    PDerr_igr = np.array([
        2.05, 2.10, 3.00, 3.90
    ])
    Eerr_igr = np.array([
        0.50, 0.50, 0.50, 1.50
    ])
    # ============================================================
    # Plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    # Colors
    color_j1727 = '#123A8C'  # dark blue
    color_cyg = '#4C78C8'  # lighter blue
    color_igr = '#8B1A1A'  # dark red
    # Swift J1727.8-1613
    ax.errorbar(
        E_j1727, PD_j1727,
        xerr=Eerr_j1727,
        yerr=PDerr_j1727,
        fmt='o',
        ms=6,
        color=color_j1727,
        ecolor=color_j1727,
        elinewidth=1.3,
        capsize=3,
        capthick=1.3,
        label='Swift J1727.8-1613'
    )
    # Cygnus X-1
    ax.errorbar(
        E_cyg, PD_cyg,
        xerr=Eerr_cyg,
        yerr=PDerr_cyg,
        fmt='o',
        ms=5,
        color=color_cyg,
        ecolor=color_cyg,
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        label='Cygnus X-1'
    )
    # IGR J17091-3624
    ax.errorbar(
        E_igr, PD_igr,
        xerr=Eerr_igr,
        yerr=PDerr_igr,
        fmt='o',
        ms=6,
        color=color_igr,
        ecolor=color_igr,
        elinewidth=1.3,
        capsize=3,
        capthick=1.3,
        label='IGR J17091-3624'
    )
    # ============================================================
    # Formatting
    # ============================================================
    ax.set_xlabel(r'Energy [keV]', fontsize=14)
    ax.set_ylabel(r'Polarization degree [\%]', fontsize=14)
    ax.set_xlim(2, 8)
    ax.set_ylim(0, 18)
    ax.set_xticks(np.arange(2, 9, 1))
    ax.set_yticks(np.arange(0, 17, 2))
    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        top=True,
        right=True,
        length=6,
        width=1.2,
        labelsize=12
    )
    ax.legend(
        fontsize=11,
        frameon=False,
        loc='upper left'
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    plt.tight_layout()
    plt.show()

def plot_soft():
    import numpy as np
    import matplotlib.pyplot as plt

    # ============================================================
    # Colors
    # ============================================================

    color_blue_dark =  '#123A8C'  # Swift J1727.8-1613
    color_blue_light = 'cyan' # GX 339-4
    color_blue_mid = '#4C78C8'  # 4U 1957+115

    color_red_dark = '#8B1A1A'  # 4U 1630-47
    color_violet = '#7B2CBF'  # GRS 1739-278

    # ============================================================
    # Data
    # ============================================================

    # ------------------------------------------------------------
    # 4U 1957+115
    # ------------------------------------------------------------

    E_1957 = np.array([2.5, 3.65, 5.15, 7.0])

    Eerr_1957 = np.array([0.5, 0.65, 0.85, 1.0])

    PD_1957 = np.array([1.6, 2.05, 3.15, 4.8])

    PDerr_1957 = np.array([0.5, 0.5, 0.9])

    # ------------------------------------------------------------
    # 4U 1630-47
    # ------------------------------------------------------------

    E_1630 = np.array([2.5, 3.5, 4.5, 5.5, 7.0])

    Eerr_1630 = np.array([0.5, 0.5, 0.5, 0.5, 1.0])

    PD_1630 = np.array([5.75, 7.20, 8.05, 9.55, 10.60])

    PDerr_1630 = np.array([0.30, 0.20, 0.25, 0.38, 0.60])

    # ------------------------------------------------------------
    # GRS 1739-278
    # ------------------------------------------------------------

    E_1739 = np.array([2.4, 3.5, 5.1, 7.0])

    Eerr_1739 = np.array([0.4, 0.7, 0.9, 1.0])

    PD_1739 = np.array([2.1, 1.5, 3.0, 10.0])

    PDerr_1739 = np.array([0.5, 0.5, 0.9, 3.0])

    # ============================================================
    # Figure
    # ============================================================

    fig, ax = plt.subplots(figsize=(7, 5))

    # ============================================================
    # Upper limits across the full IXPE band
    # ============================================================

    # GX 339-4: PD < 1.2%
    ax.hlines(
        y=1.2,
        xmin=2.0,
        xmax=8.0,ls='--',
        color=color_blue_light,
        linewidth=1.5,
        label='GX 339-4 upper limit'
    )

    ax.plot(
        5.0, 1.2,
        marker='v',
        markersize=7,
        color=color_blue_light
    )

    # Swift J1727.8-1613: PD < 1.34%
    ax.hlines(
        y=1.34,
        xmin=2.0,
        xmax=8.0,ls='--',
        color=color_blue_dark,
        linewidth=1.5,
        label='Swift J1727.8-1613 upper limit'
    )

    ax.plot(
        5.0, 1.34,
        marker='v',
        markersize=7,
        color=color_blue_dark
    )

    # ============================================================
    # 4U 1957+115
    # ============================================================

    # First three measurements
    ax.errorbar(
        E_1957[:3],
        PD_1957[:3],
        xerr=Eerr_1957[:3],
        yerr=PDerr_1957,
        fmt='o',
        ms=5.5,
        color=color_blue_mid,
        ecolor=color_blue_mid,
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        label='4U 1957+115'
    )

    # Last point: upper limit < 4.8%
    ax.errorbar(
        E_1957[3],
        PD_1957[3],
        xerr=Eerr_1957[3],
        fmt='none',
        ecolor=color_blue_mid,
        elinewidth=1.2,
        capsize=3
    )

    ax.plot(
        E_1957[3],
        PD_1957[3],
        marker='v',ls='--',
        markersize=7,
        color=color_blue_mid
    )

    # ============================================================
    # 4U 1630-47
    # ============================================================

    ax.errorbar(
        E_1630,
        PD_1630,
        xerr=Eerr_1630,
        yerr=PDerr_1630,
        fmt='o',
        ms=5.5,
        color=color_red_dark,
        ecolor=color_red_dark,
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        label='4U 1630-47'
    )

    # ============================================================
    # GRS 1739-278
    # ============================================================

    ax.errorbar(
        E_1739,
        PD_1739,
        xerr=Eerr_1739,
        yerr=PDerr_1739,
        fmt='o',
        ms=5.5,
        color=color_violet,
        ecolor=color_violet,
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        label='GRS 1739-278'
    )

    # ============================================================
    # Formatting
    # ============================================================

    ax.set_xlabel(r'Energy [keV]', fontsize=14)
    ax.set_ylabel(r'Polarization degree [\%]', fontsize=14)

    ax.set_xlim(2, 8)
    ax.set_ylim(0, 18)

    ax.set_xticks(np.arange(2, 9, 1))
    ax.set_yticks(np.arange(0, 15, 2))

    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        top=True,
        right=True,
        length=6,
        width=1.2,
        labelsize=12
    )

    ax.legend(
        fontsize=10,
        frameon=False,
        loc='upper left'
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.show()

    pass

def yetanotherweirdparser(filename_init):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import label

    # ============================================================
    # USER SETTINGS
    # ============================================================

    filename=filename_init

    E_LEFT = 6.4
    E_RIGHT = 8.4

    # ============================================================
    # LOAD IMAGE
    # ============================================================

    img = np.asarray(Image.open(filename).convert("RGB"))

    H, W, _ = img.shape

    print(f"Image size: {W} x {H} pixels")

    # ============================================================
    # MAKE A BLACK-PIXEL MASK
    # ============================================================

    # Black axes are substantially darker than the colored curves.
    gray = (
            0.299 * img[:, :, 0] +
            0.587 * img[:, :, 1] +
            0.114 * img[:, :, 2]
    )

    black = gray < 60

    # ============================================================
    # FIND THE PLOT FRAME
    # ============================================================

    # A plot border produces a very long uninterrupted sequence
    # of dark pixels.  We therefore look for columns/rows having
    # unusually long consecutive runs of black pixels.

    def longest_run_1d(a):
        """Length of longest consecutive True run."""
        if not np.any(a):
            return 0

        x = np.concatenate(([False], a, [False])).astype(np.int8)
        changes = np.diff(x)

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        return np.max(ends - starts)

    # ------------------------------------------------------------
    # Vertical black edges
    # ------------------------------------------------------------

    vertical_run = np.array([
        longest_run_1d(black[:, x])
        for x in range(W)
    ])

    # Ignore a small margin at the extreme image boundaries.
    margin_x = max(5, int(0.01 * W))

    candidate_x = np.arange(margin_x, W - margin_x)

    # The plot's vertical axes should have very long black runs.
    # Require a run spanning at least ~50% of the image height.
    long_vertical = candidate_x[
        vertical_run[candidate_x] > 0.50 * H
        ]

    if len(long_vertical) < 2:
        raise RuntimeError(
            "Could not automatically find the left/right plot edges. "
            "Try increasing the black threshold from 60 to 80."
        )

    # Usually there are a few adjacent pixels belonging to each
    # axis line. Group them into clusters.
    groups = np.split(
        long_vertical,
        np.where(np.diff(long_vertical) > 1)[0] + 1
    )

    # Keep substantial groups
    groups = [g for g in groups if len(g) > 0]

    # The outermost groups correspond to the plot frame.
    left_edge = int(round(np.mean(groups[0])))
    right_edge = int(round(np.mean(groups[-1])))

    print(f"Detected plot edges: x = {left_edge}, {right_edge}")

    # ============================================================
    # FIND THE BOTTOM PLOT AXIS (y = 0)
    # ============================================================

    horizontal_run = np.array([
        longest_run_1d(black[y, :])
        for y in range(H)
    ])

    candidate_y = np.arange(0, H)

    long_horizontal = candidate_y[
        horizontal_run[candidate_y] > 0.50 * W
        ]

    if len(long_horizontal) == 0:
        raise RuntimeError("Could not find the bottom/top plot edges.")

    groups_y = np.split(
        long_horizontal,
        np.where(np.diff(long_horizontal) > 1)[0] + 1
    )

    groups_y = [g for g in groups_y if len(g) > 0]

    # The bottom border is the lowest long horizontal line.
    bottom_edge = int(round(np.mean(groups_y[-1])))

    print(f"Detected bottom edge: y = {bottom_edge}")

    # ============================================================
    # FIND y = 1.0
    # ============================================================

    # The y-axis has tick marks at 0.0, 0.2, ..., 1.0.
    #
    # We inspect a small region immediately to the right of the
    # left axis and look for horizontal black tick marks.

    search_width = max(20, int(0.04 * W))

    x_start = left_edge
    x_stop = min(W, left_edge + search_width)

    # Number of black pixels in this region at each row
    tick_score = black[:, x_start:x_stop].sum(axis=1)

    # A tick is a horizontal feature, so it produces several
    # black pixels on one row.
    threshold = max(3, int(0.15 * search_width))

    tick_rows = np.where(tick_score >= threshold)[0]

    # Group adjacent rows belonging to the same tick.
    if len(tick_rows) == 0:
        raise RuntimeError("Could not detect y-axis tick marks.")

    tick_groups = np.split(
        tick_rows,
        np.where(np.diff(tick_rows) > 1)[0] + 1
    )

    tick_groups = [
        g for g in tick_groups
        if len(g) > 0
    ]

    tick_positions = np.array([
        np.mean(g) for g in tick_groups
    ])

    print("Detected y tick positions:")
    print(tick_positions)

    # ------------------------------------------------------------
    # Select the tick corresponding to y = 1.0.
    #
    # It is the highest tick below the top border.
    # ------------------------------------------------------------

    # Exclude anything essentially at the top image boundary.
    valid_ticks = tick_positions[
        tick_positions > 0.05 * H
        ]

    if len(valid_ticks) == 0:
        raise RuntimeError("Could not identify y=1.0 tick.")

    # The y=1.0 tick should be the highest one in the plot.
    y_one = valid_ticks[0]

    print(f"Detected y=1.0 at pixel y = {y_one:.2f}")

    # ============================================================
    # PIXEL -> PHYSICAL COORDINATE CALIBRATION
    # ============================================================

    # We use the CENTER of the detected black border as the
    # coordinate boundary.

    x_pixels = np.arange(left_edge, right_edge + 1)

    energy = (
            E_LEFT
            + (x_pixels - left_edge)
            / (right_edge - left_edge)
            * (E_RIGHT - E_LEFT)
    )

    # y increases downwards in an image.
    #
    # bottom_edge -> y = 0
    # y_one       -> y = 1
    #
    # Therefore:
    #
    # yphysical = (bottom - ypixel) / (bottom - y_one)

    def pixel_to_y(y_pixel):
        return (
                bottom_edge - y_pixel
        ) / (
                bottom_edge - y_one
        )

    # ============================================================
    # ROBUST COLOR SEGMENTATION
    # ============================================================

    # Work only inside the detected plotting rectangle.
    plot = img[:, left_edge:right_edge + 1, :].astype(np.float32)

    R = plot[:, :, 0]
    G = plot[:, :, 1]
    B = plot[:, :, 2]

    # ------------------------------------------------------------
    # RED
    #
    # Red curve should have R substantially larger than G and B.
    # We deliberately use fairly loose thresholds.
    # ------------------------------------------------------------

    red_mask = (
            (R > 120) &
            (R > 1.25 * G) &
            (R > 1.25 * B) &
            ((R - G) > 50) &
            ((R - B) > 50)
    )

    # ------------------------------------------------------------
    # YELLOW / ORANGE
    #
    # Yellow/orange has:
    #   R high
    #   G reasonably high
    #   B substantially lower
    # ------------------------------------------------------------

    yellow_mask = (
            (R > 120) &
            (G > 70) &
            (R > 1.15 * G) &
            (G > 1.25 * B) &
            ((R - B) > 60)
    )

    # ============================================================
    # IMPORTANT DIAGNOSTIC
    # ============================================================

    print()
    print("Color-pixel diagnostics:")
    print("  Red pixels:   ", np.sum(red_mask))
    print("  Yellow pixels:", np.sum(yellow_mask))

    if np.sum(red_mask) == 0:
        raise RuntimeError(
            "No red pixels detected. Print some RGB values from "
            "the curve and adjust the red thresholds."
        )

    if np.sum(yellow_mask) == 0:
        raise RuntimeError(
            "No yellow/orange pixels detected. Print some RGB values "
            "from the curve and adjust the yellow thresholds."
        )

    # ============================================================
    # EXTRACT CURVE
    # ============================================================

    def extract_curve(color_mask, x_pixels):
        """
        Extract the colored curve from a binary color mask.

        For each x column:
          - find all colored pixels
          - select the pixel closest to the expected curve position
          - use the median when several pixels belong to the curve
        """

        y_pixels = np.full(len(x_pixels), np.nan)

        for i, x in enumerate(x_pixels):

            # x is in full-image coordinates, whereas color_mask
            # starts at left_edge.
            xx = x - left_edge

            ys = np.where(color_mask[:, xx])[0]

            if len(ys) == 0:
                continue

            # Only accept pixels inside the plotting region.
            ys = ys[
                (ys >= int(y_one) - 10) &
                (ys <= bottom_edge + 2)
                ]

            if len(ys) == 0:
                continue

            # If multiple colored pixels occur in a column, their
            # median is a good estimate of the center of the line.
            y_pixels[i] = np.median(ys)

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------

        good = np.isfinite(y_pixels)

        print(
            f"Detected {good.sum()} / {len(y_pixels)} "
            f"x-columns ({100 * good.mean():.1f}%)"
        )

        if good.sum() < 2:
            raise RuntimeError(
                "Too few curve pixels detected."
            )

        # --------------------------------------------------------
        # Fill gaps caused by anti-aliasing or overlap with other
        # curves.
        # --------------------------------------------------------

        y_pixels = np.interp(
            np.arange(len(y_pixels)),
            np.flatnonzero(good),
            y_pixels[good]
        )

        return pixel_to_y(y_pixels)

    # ============================================================
    # EXTRACT RED AND YELLOW
    # ============================================================

    red_y = extract_curve(red_mask, x_pixels)
    yellow_y = extract_curve(yellow_mask, x_pixels)

    # ============================================================
    # FINAL ARRAYS
    # ============================================================

    red = np.column_stack((energy, red_y))
    yellow = np.column_stack((energy, yellow_y))

    np.set_printoptions(
        precision=7,
        suppress=True,
        linewidth=200
    )

    print("\nRED:")
    print(red)

    print("\nYELLOW:")
    print(yellow)

    # ============================================================
    # SAVE
    # ============================================================

    np.save("red.npy", red)
    np.save("yellow.npy", yellow)

    np.savetxt(
        "red.csv",
        red,
        delimiter=",",
        header="energy_keV,red_fraction",
        comments=""
    )

    np.savetxt(
        "yellow.csv",
        yellow,
        delimiter=",",
        header="energy_keV,yellow_fraction",
        comments=""
    )


import numpy as np
from scipy.signal import savgol_filter


def robust_smooth(
    data,
    window=21,
    polynomial=3,
    outlier_window=15,
    outlier_sigma=4
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    """
    Robust smoothing for curves digitized from images.

    1. Detects isolated pixel-extraction errors.
    2. Interpolates those points.
    3. Applies Savitzky-Golay smoothing.
    """

    x = data[:, 0].copy()
    y = data[:, 1].copy()

    # --------------------------------------------------------
    # First-pass smooth curve
    # --------------------------------------------------------

    if outlier_window % 2 == 0:
        outlier_window += 1

    baseline = savgol_filter(
        y,
        outlier_window,
        polynomial
    )

    # --------------------------------------------------------
    # Residual from smooth local trend
    # --------------------------------------------------------

    residual = y - baseline

    # Robust estimate of noise using MAD
    median_residual = np.median(residual)

    mad = np.median(
        np.abs(residual - median_residual)
    )

    # Convert MAD to approximately standard deviation
    robust_sigma = 1.4826 * mad

    if robust_sigma == 0:
        robust_sigma = np.std(residual)

    # --------------------------------------------------------
    # Identify isolated bad pixels
    # --------------------------------------------------------

    bad = (
        np.abs(residual - median_residual)
        > outlier_sigma * robust_sigma
    )

    print(
        f"Removed {bad.sum()} suspected "
        f"outlier points out of {len(y)}"
    )

    # --------------------------------------------------------
    # Interpolate bad points
    # --------------------------------------------------------

    good = ~bad

    y_clean = y.copy()

    if good.sum() >= 2:
        y_clean[bad] = np.interp(
            x[bad],
            x[good],
            y[good]
        )

    # --------------------------------------------------------
    # Final smoothing
    # --------------------------------------------------------

    if window % 2 == 0:
        window += 1

    if window >= len(y_clean):
        window = len(y_clean) - 1
        if window % 2 == 0:
            window -= 1

    y_smooth = savgol_filter(
        y_clean,
        window,
        polynomial
    )

    return np.column_stack((x, y_smooth))

#
# # ============================================================
# # APPLY
# # ============================================================
#
# red_smooth = robust_smooth(
#     red,
#     window=21,
#     polynomial=3
# )
#
# yellow_smooth = robust_smooth(
#     yellow,
#     window=21,
#     polynomial=3
# )
# yellow_smooth_11 = robust_smooth(yellow, window=11)
# yellow_smooth_21 = robust_smooth(yellow, window=21)
# yellow_smooth_31 = robust_smooth(yellow, window=31)


def combi_sp_plot_soft():

    import os
    from xspec_config_multisp import plot_comp_ratio
    import matplotlib.pyplot as plt
    os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/photo/pion/comp_plots')
    plot_comp_ratio([1, 2], [[1, 2, 3], [1, 2], [1, 2], [1, 2], [3]], 6.38, 7.13,
                    other_addcomps_labels=['combined', 'photo 1',
                                           'photo 2',
                                           'photo 3',
                                           'photo 1 em',
                                           ],
                    other_addcomps_colors=['darkred', 'red', 'red', 'red', ],
                    other_addcomps_alpha=[1, 0.5, 0.5, 0.5, 0.5],
                    other_addcomps_type=['abs', 'abs', 'abs', 'abs', 'abs'],
                    other_addcomps_ls=['-', '--', 'dashdot', ':'],
                    cont_addcomps_xcm='3comp1em_closersol_2-10_deabs_cont.xcm',
                    other_addcomps_xcm=['3comp1em_closersol_2-10_deabs_tot.xcm',
                                        '3comp1em_closersol_2-10_deabs_tot_abs1.xcm',
                                        '3comp1em_closersol_2-10_deabs_tot_abs2.xcm',
                                        '3comp1em_closersol_2-10_deabs_tot_abs3.xcm',
                                        '3comp1em_closersol_2-10_deabs_tot.xcm', ],
                    ylims=[0.01, 1.], figsize=(6, 4), minor_locator=10, ylabel_prefix='',plot_transi=False)

    ax=plt.gca()

    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/Propal/IXPE/AO4/windyBH/fig_spe/arXiv-2506.07319v1')
    red_smooth_31 = np.loadtxt('red_smooth_31.npy')
    yellow_smooth_31 = np.loadtxt('yellow_smooth_31.npy')

    red_below_6p74 = np.array([
    [6.5225, 0.986],
    [6.5245, 0.986],
    [6.5265, 0.986],
    [6.5285, 0.986],
    [6.5305, 0.986],
    [6.5325, 0.986],
    [6.5345, 0.986],
    [6.5365, 0.986],
    [6.5385, 0.986],
    [6.5405, 0.986],
    [6.5425, 0.986],
    [6.5445, 0.986],
    [6.5465, 0.986],
    [6.5485, 0.986],
    [6.5505, 0.986],
    [6.5525, 0.986],
    [6.5545, 0.986],
    [6.5565, 0.986],
    [6.5585, 0.986],
    [6.5605, 0.986],
    [6.5625, 0.986],
    [6.5645, 0.986],
    [6.5665, 0.986],
    [6.5685, 0.986],
    [6.5705, 0.986],
    [6.5725, 0.986],
    [6.5745, 0.986],
    [6.5765, 0.986],
    [6.5785, 0.986],
    [6.5805, 0.986],
    [6.5825, 0.986],
    [6.5845, 0.986],
    [6.5865, 0.986],
    [6.5885, 0.986],
    [6.5905, 0.986],
    [6.5925, 0.986],
    [6.5945, 0.986],
    [6.5965, 0.986],
    [6.5985, 0.986],
    [6.6005, 0.986],
    [6.6025, 0.986],
    [6.6045, 0.986],
    [6.6065, 0.986],
    [6.6085, 0.986],
    [6.6105, 0.986],
    [6.6125, 0.986],
    [6.6145, 0.986],
    [6.6165, 0.986],
    [6.6185, 0.986],
    [6.6205, 0.986],
    [6.6225, 0.986],
    [6.6245, 0.986],
    [6.6265, 0.986],
    [6.6285, 0.986],
    [6.6305, 0.986],
    [6.6325, 0.986],
    [6.6345, 0.986],
    [6.6365, 0.986],
    [6.6385, 0.986],
    [6.6405, 0.986],
    [6.6425, 0.986],
    [6.6445, 0.986],
    [6.6465, 0.986],
    [6.6485, 0.986],
    [6.6505, 0.986],
    [6.6525, 0.986],
    [6.6545, 0.986],
    [6.6565, 0.986],
    [6.6585, 0.986],
    [6.6605, 0.986],
    [6.6625, 0.986],
    [6.6645, 0.986],
    [6.6665, 0.985],
    [6.6685, 0.984],
    [6.6705, 0.981],
    [6.6725, 0.976],
    [6.6745, 0.966],
    [6.6765, 0.955],
    [6.6785, 0.948],
    [6.6805, 0.946],
    [6.6825, 0.950],
    [6.6845, 0.957],
    [6.6865, 0.965],
    [6.6885, 0.969],
    [6.6905, 0.966],
    [6.6925, 0.944],
    [6.6945, 0.870],
    [6.6965, 0.760],
    [6.6985, 0.704],
    [6.7005, 0.688],
    [6.7025, 0.695],
    [6.7045, 0.735],
    [6.7065, 0.800],
    [6.7085, 0.858],
    [6.7105, 0.904],
    [6.7125, 0.939],
    [6.7145, 0.960],
    [6.7165, 0.974],
    [6.7185, 0.981],
    [6.7205, 0.984],
    [6.7225, 0.985],
    [6.7245, 0.986],
    [6.7265, 0.986],
    [6.7285, 0.986],
    [6.7305, 0.986],
    [6.7325, 0.986],
    [6.7345, 0.986]
])

    red_smooth_use=np.array(red_smooth_31[red_smooth_31.T[0]<6.53].tolist()+red_below_6p74.tolist()+red_smooth_31[red_smooth_31.T[0]>6.7345].tolist())

    yellow_smooth_use=yellow_smooth_31.T

    yellow_smooth_use[1][yellow_smooth_use[0]>6.72]=yellow_smooth_31[-1][1]

    yellow_smooth_use[1][543:566]=yellow_smooth_use[1][520:543][::-1]
    yellow_smooth_use[1][566:600]=np.repeat(yellow_smooth_use[1][600],34)
    yellow_smooth_use=yellow_smooth_use.T

    #readjusting the false interpolation above 6.70 by reflecting the left part of the gaussian profile


    plt.plot(red_smooth_use.T[0],red_smooth_use.T[1]/100,color='red',ls='--',alpha=0.5)
    plt.plot(yellow_smooth_use.T[0],yellow_smooth_use.T[1]/100,color='red',ls='--',alpha=0.5)

    def interpolate_common_x_union(arr1, arr2):
        x1, y1 = arr1[:, 0], arr1[:, 1]
        x2, y2 = arr2[:, 0], arr2[:, 1]

        # Only use the region where both curves exist
        xmin = max(x1.min(), x2.min())
        xmax = min(x1.max(), x2.max())

        # Union of both x grids within overlap
        x_common = np.unique(np.concatenate([
            x1[(x1 >= xmin) & (x1 <= xmax)],
            x2[(x2 >= xmin) & (x2 <= xmax)]
        ]))

        y1_interp = np.interp(x_common, x1, y1)
        y2_interp = np.interp(x_common, x2, y2)

        return (
            np.column_stack((x_common, y1_interp)),
            np.column_stack((x_common, y2_interp))
        )


    red_common_sampl, yellow_common_sampl = interpolate_common_x_union(red_smooth_use, yellow_smooth_use)

    plt.plot(red_common_sampl.T[0],red_common_sampl.T[1]*yellow_common_sampl.T[1]/100,color='darkred',ls='-',)

    plt.axhline(2,color='darkblue')

    plt.yscale('log')
    plt.ylim(3e-3, 3)

    plt.xlim(6.4,7.05)

    plt.text(6.75,4e-3,r'4U 1630-47'+'\n'+r'$\sim3\%$ $L_{Edd}$',color='darkred')

    plt.text(6.75, 3e-1, r'MAXI J1543-564'+'\n'+r'$\sim10\%$ $L_{Edd}$', color='darkred')

    plt.text(6.55, 1.4, r'4U 1957+115, Swift J1727.8-1613, GX 339-4', color='midnightblue')

    plt.gca().get_legend().remove()

    plt.ylabel('normalized ratios to continuum')

def combi_sp_plot_hard():

    import os
    from xspec_config_multisp import plot_comp_ratio
    import matplotlib.pyplot as plt
    os.chdir('/media/parrazyte/crucial_SSD/Observ/BHLMXB/XRISM/MAXIJ1543-564/photo/pion/comp_plots')
    plot_comp_ratio([1, 2], [[1, 2, 3], [1, 2], [1, 2], [1, 2], [3]], 6.38, 7.13,
                    other_addcomps_labels=['combined', 'photo 1',
                                           'photo 2',
                                           'photo 3',
                                           'photo 1 em',
                                           ],
                    other_addcomps_colors=['white','white','white','white', ],
                    other_addcomps_alpha=[1, 0.5, 0.5, 0.5, 0.5],
                    other_addcomps_type=['abs', 'abs', 'abs', 'abs', 'abs'],
                    other_addcomps_ls=['-', '--', 'dashdot', ':'],
                    cont_addcomps_xcm='3comp1em_closersol_2-10_deabs_cont.xcm',
                    other_addcomps_xcm=['3comp1em_closersol_2-10_deabs_tot.xcm',
                                        '3comp1em_closersol_2-10_deabs_tot_abs1.xcm',
                                        '3comp1em_closersol_2-10_deabs_tot_abs2.xcm',
                                        '3comp1em_closersol_2-10_deabs_tot_abs3.xcm',
                                        '3comp1em_closersol_2-10_deabs_tot.xcm', ],
                    ylims=[0.01, 1.], figsize=(6, 4), minor_locator=10, ylabel_prefix='',plot_transi=False)

    ax=plt.gca()

    os.chdir('/home/parrazyte/Documents/Work/PostDoc/docs/Propal/IXPE/AO4/windyBH/fig_spe/arXiv-2506.07319v1')


    plt.axhline(3e-1,color='darkblue')

    plt.axhline(3e-2,color='darkred')

    plt.yscale('log')
    plt.ylim(3e-3, 3)

    plt.xlim(6.4,7.05)

    plt.text(6.75,1.5e-2,r'IGR J17091-3624',color='darkred')

    plt.text(6.75, 1.5e-1, r'GX 339-4', color='midnightblue')

    plt.gca().get_legend().remove()

    plt.ylabel('normalized ratios to continuum')