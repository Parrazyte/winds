

# 10x the base flux for 0p1Edd 8Msun 8kpc, to get 50ks in this setup
def AMD_det(figsize=(6,5),
            #for 50ks at 0.1LEdd
            flux_lim_NA=7.824e-08,
            plot_AMD_photo=True):
    '''
    wrapper to reproduce and add elements to the AMDs of Keshet et al. 2025, 2026
    '''

    import os
    import numpy as np
    from xspec import Xset, AllModels
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib.lines import Line2D

    # from Keshet25, digitized by ChatGPT
    # logxi, logNH

    # new integrated figure with weird results
    # amd_GROJ = np.array([
    #     [1.10, 15.4],
    #     [2.95, 21.7],
    #     [3.30, 22.5],
    #     [3.45, 22.5],
    #     [5.00, 23.4],
    # ]).T
    #
    # old figure
    amd_GROJ = np.array([
        [1.10, 16.4],
        [2.95, 22.85],
        [3.30, 23.6],
        [5.00, 23.5],
    ]).T

    # full values without actual bounds for the non-probed parts of the AMD
    # amd_GRS_soft = np.array([
    #     [0.00, 17.20],
    #     [6.50, 23.70],
    # ]).T
    #
    # amd_GX = np.array([
    #     [0.00, 17.9],
    #     [5.40, 23.30],
    # ]).T
    #
    # amd_4U = np.array([
    #     [0.00, 18.70],
    #     [5.40, 24.10],
    # ]).T

    # new integrated figure with weird results
    # amd_GRS_soft = np.array([
    #     [2.4, 17.20+2.4],
    #     [5.7, 17.20+5.7],
    # ]).T

    amd_GX = np.array([
        [2.4, 17.85 + 2.4],
        [5.40, 23.30],
    ]).T

    amd_4U = np.array([
        [2.4, 18.70 + 2.4],
        [5.40, 24.10],
    ]).T

    # logNH_GRS1915 = logxi + 17.20
    # logNH_GX13 = logxi + 17.85
    # logNH_4U1630 = logxi + 18.70

    # from Keshet et al. 26, digitized by ChatGPT and verified with automeris.io
    # amd_GRS_hard1 = np.array([
    #     [1.00, 18.70],
    #     [3.0, 22.70],
    #     [3.3, 22.70],
    #     [4.00, 22.00],
    # ]).T
    #
    # amd_GRS_hard2 = np.array([
    #     [1.00, 20.40],
    #     [3.40, 22.80],
    #     [4.0, 22.20],
    # ]).T

    # cropped version starting at the logxi peak of Si12+

    amd_GRS_hard1 = np.array([
        [2.00, 21.20],
        [3.0, 22.70],
        [3.3, 22.70],
        [4.00, 22.00],
    ]).T

    amd_GRS_hard2 = np.array([
        [2.00, 21.40],
        [3.40, 22.80],
        [4.0, 22.20],
    ]).T

    # amd_GRS_soft= np.array([
    #     [2.00, 18.50],
    #     [6.50, 23.00],
    # ]).T

    amd_GRS_soft = np.array([
        [2.00, 18.50],
        [5.7, 22.3],
    ]).T

    fig, ax = plt.subplots(1, figsize=figsize)

    # =====================================================================
    # Plotting function (asymmetric errors)
    # =====================================================================

    def plot_nh_xi(ax, NH, NH_err_lo, NH_err_hi,
                   x, x_err_lo, x_err_hi,
                   marker, color,
                   NH2=None, NH_err_lo2=None, NH_err_hi2=None,
                   x2=None, x_err_lo2=None, x_err_hi2=None,
                   color2=None, two_zones=False):
        """
        Plot log10(N_H) + 22 vs. log(xi) as an errorbar series, with
        asymmetric (lo, hi) uncertainties on both axes.

        If two_zones=True, the input arrays contain zone 1 followed by
        zone 2. Alternatively, zone 2 can be supplied through the *2
        arguments.
        """

        ln10 = np.log(10)

        if two_zones:
            n = len(NH) // 2

            NH1 = NH[:n]
            NH2 = NH[n:]
            NH_err_lo1 = NH_err_lo[:n]
            NH_err_lo2 = NH_err_lo[n:]
            NH_err_hi1 = NH_err_hi[:n]
            NH_err_hi2 = NH_err_hi[n:]

            x1 = x[:n]
            x2 = x[n:]
            x_err_lo1 = x_err_lo[:n]
            x_err_lo2 = x_err_lo[n:]
            x_err_hi1 = x_err_hi[:n]
            x_err_hi2 = x_err_hi[n:]

            if color2 is None:
                color2 = color

        else:
            NH1 = NH
            NH_err_lo1 = NH_err_lo
            NH_err_hi1 = NH_err_hi
            x1 = x
            x_err_lo1 = x_err_lo
            x_err_hi1 = x_err_hi

        y1 = np.log10(NH1) + 22
        y_err_lo1 = NH_err_lo1 / (NH1 * ln10)
        y_err_hi1 = NH_err_hi1 / (NH1 * ln10)

        ax.errorbar(
            x1, y1,
            xerr=np.array([x_err_lo1, x_err_hi1]),
            yerr=np.array([y_err_lo1, y_err_hi1]),
            marker=marker, color='None',
            ecolor=color, markerfacecolor='None',
            markeredgecolor=color, alpha=0.5
        )

        if NH2 is not None:
            y2 = np.log10(NH2) + 22
            y_err_lo2 = NH_err_lo2 / (NH2 * ln10)
            y_err_hi2 = NH_err_hi2 / (NH2 * ln10)

            ax.errorbar(
                x2, y2,
                xerr=np.array([x_err_lo2, x_err_hi2]),
                yerr=np.array([y_err_lo2, y_err_hi2]),
                marker=marker, color='None',
                ecolor=color2, markerfacecolor='None',
                markeredgecolor=color2, alpha=0.5
            )

            if plot_AMD_photo:
                for i in range(min(len(x1), len(x2))):
                    xm = (x1[i] + x2[i]) / 2
                    ym = (y1[i] + y2[i]) / 2

                    ax.plot(
                        [x1[i], xm], [y1[i], ym],
                        color=color, alpha=0.1, zorder=0
                    )
                    ax.plot(
                        [xm, x2[i]], [ym, y2[i]],
                        color=color2, alpha=0.1, zorder=0
                    )

    plt.xlabel(r'log$\xi$')
    plt.ylabel(r'dlogN$_H$/dlog$\xi$')
    plt.plot(amd_GROJ[0], amd_GROJ[1], color='darkorange', label='', ls=':', alpha=0.5)

    plt.plot(amd_GRS_soft[0], amd_GRS_soft[1], color='red', ls=':', label='', alpha=0.5)
    plt.plot(amd_GX[0], amd_GX[1], color='red', label='', ls=':', alpha=0.5)
    plt.plot(amd_4U[0], amd_4U[1], color='red', label='', ls=':', alpha=0.5)

    plt.plot(amd_GRS_hard1[0], amd_GRS_hard1[1], color='blue', ls=':', label='', alpha=0.5)
    plt.plot(amd_GRS_hard2[0], amd_GRS_hard2[1], color='blue', label='', ls=':', alpha=0.5)

    plt.scatter(amd_GROJ[0], amd_GROJ[1], facecolor='grey',color='darkorange', label='',
                ls='None', marker='o', alpha=0.5)

    plt.scatter(amd_GRS_soft[0], amd_GRS_soft[1],facecolor='grey',edgecolor='red', label='',
                ls='None', marker='D', alpha=0.5)
    plt.scatter(amd_GX[0], amd_GX[1],facecolor='grey',edgecolor='red', label='',
                ls='None', marker='X', alpha=0.5)
    plt.scatter(amd_4U[0], amd_4U[1],facecolor='grey',edgecolor='red', label='',
                ls='None', marker='s', alpha=0.5)

    plt.scatter(amd_GRS_hard1[0], amd_GRS_hard1[1], facecolor='grey',edgecolor='blue', label='',
                ls='None', marker='D', alpha=0.5)
    plt.scatter(amd_GRS_hard2[0], amd_GRS_hard2[1], facecolor='grey',edgecolor='blue', label='', ls='None',
                marker='D', alpha=0.5)

    plt.plot([], [], color='orange', lw=6, label='Super-Eddington')
    plt.plot([], [], color='red', lw=6, label='soft')
    plt.plot([], [], color='magenta', lw=6, label='intermediate')
    plt.plot([], [], color='blue', lw=6, label='obscured')

    plt.errorbar([], [], marker='s', color='None',label='photoionization fits\n(discrete)',
                 ecolor='black', markerfacecolor='None', markeredgecolor='black',)
    plt.scatter([],[], facecolor='grey',edgecolor='black',
                label='line by line\n(continuous)', color='black', marker='s',)

    plt.legend(loc='upper left')

    import numpy as np
    import matplotlib.pyplot as plt

    ln10 = np.log(10)

    # =====================================================================
    # Your (paper-extracted) arrays, with asymmetric (lo, hi) uncertainties,
    # pulled directly from the table built earlier in this conversation.
    # Zones restricted to |v_abs| < 10e-1 c, uppermost first-order model
    # in each source's table (Tables 7-10).
    # Miller+15
    # =====================================================================

    # --- GRO J1655-40 (model 1655-1a, zones 1-2) ---
    NH_1655 = np.array([59.0, 8.2])
    NH_err_lo_1655 = np.array([1.0, 0.5])
    NH_err_hi_1655 = np.array([3.0, 1.2])
    logxi_1655 = np.array([4.72, 4.53])
    logxi_err_lo_1655 = np.array([0.04, 0.03])
    logxi_err_hi_1655 = np.array([0.04, 0.03])

    # --- GRS 1915+105 (model 1915-1a, zones 1-3) ---
    #we only take the first 2 zones to make things simpler, even if the third zone is "only" at 2000km/s
    NH_1915 = np.array([42.0, 60.0, 0.7])[:-1]
    NH_err_lo_1915 = np.array([4.0, 10.0, 0.1])[:-1]
    NH_err_hi_1915 = np.array([4.0, 10.0, 0.2])[:-1]
    logxi_1915 = np.array([3.76, 5.04, 3.82])[:-1]
    logxi_err_lo_1915 = np.array([0.03, 0.06, 0.04])[:-1]
    logxi_err_hi_1915 = np.array([0.03, 0.2, 0.04])[:-1]

    #removed because it's alrady in the Trueba+19
    # # --- 4U 1630-472 (model 1630-1a, zones 1-2) ---
    # NH_1630 = np.array([22.0, 7.0])
    # NH_err_lo_1630 = np.array([2.0, 2.0])
    # NH_err_hi_1630 = np.array([2.0, 3.0])
    # logxi_1630 = np.array([4.14, 4.6])
    # logxi_err_lo_1630 = np.array([0.02, 0.3])
    # logxi_err_hi_1630 = np.array([0.02, 0.1])

    # --- H 1743-122 (model 1743-1a, zones 1-2) ---
    NH_1743 = np.array([6.1, 14.0])
    NH_err_lo_1743 = np.array([0.6, 3.0])
    NH_err_hi_1743 = np.array([0.6, 8.0])
    logxi_1743 = np.array([4.57, 6.0])
    logxi_err_lo_1743 = np.array([0.03, 0.5])
    logxi_err_hi_1743 = np.array([0.03, 0.0])

    # GRO J1655-40
    plot_nh_xi(
        ax, NH_1655, NH_err_lo_1655, NH_err_hi_1655,
        logxi_1655, logxi_err_lo_1655, logxi_err_hi_1655,
        marker='o', color='orange', two_zones=True
    )

    # GRS 1915+105
    plot_nh_xi(
        ax, NH_1915, NH_err_lo_1915, NH_err_hi_1915,
        logxi_1915, logxi_err_lo_1915, logxi_err_hi_1915,
        marker='D', color='red',two_zones=True
    )

    # # 4U 1630-472 (paper)
    # plot_nh_xi(ax, NH_1630, NH_err_lo_1630, NH_err_hi_1630,
    #            logxi_1630, logxi_err_lo_1630, logxi_err_hi_1630,
    #            marker='s', color='red')

    # H 1743-122
    plot_nh_xi(
        ax, NH_1743, NH_err_lo_1743, NH_err_hi_1743,
        logxi_1743, logxi_err_lo_1743, logxi_err_hi_1743,
        marker='<', color='red', two_zones=True
    )

    # =====================================================================
    # Miller+16 GRS 1915
    # =====================================================================

    NH_GRS_Miller16 = np.array([30.0])
    NH_err_lo_GRS_Miller16 = np.array([2.0])
    NH_err_hi_GRS_Miller16 = np.array([2.0])
    x_GRS_Miller16 = np.array([4.04])
    x_err_lo_GRS_Miller16 = np.array([0.02])
    x_err_hi_GRS_Miller16 = np.array([0.02])

    NH2_GRS_Miller16 = np.array([0.65])
    NH_err_lo2_GRS_Miller16 = np.array([0.05])
    NH_err_hi2_GRS_Miller16 = np.array([0.05])
    x2_GRS_Miller16 = np.array([3.87])
    x_err_lo2_GRS_Miller16 = np.array([0.05])
    x_err_hi2_GRS_Miller16 = np.array([0.05])

    plot_nh_xi(
        ax,
        NH_GRS_Miller16, NH_err_lo_GRS_Miller16, NH_err_hi_GRS_Miller16,
        x_GRS_Miller16, x_err_lo_GRS_Miller16, x_err_hi_GRS_Miller16,
        marker='D', color='red',
        NH2=NH2_GRS_Miller16,
        NH_err_lo2=NH_err_lo2_GRS_Miller16,
        NH_err_hi2=NH_err_hi2_GRS_Miller16,
        x2=x2_GRS_Miller16,
        x_err_lo2=x_err_lo2_GRS_Miller16,
        x_err_hi2=x_err_hi2_GRS_Miller16,
        color2='red'
    )

    # =====================================================================
    # Trueba+19
    # =====================================================================

    # # --- 4U 1630-472 (other source) ---
    # N_H_4U = np.array([54, 14.4, 43.5, 17.6, 38.4, 12.4, 56, 21.3, 23.7, 3.5, 7.7])
    # N_H_lo_4U = np.array([17, 3.0, 17, 3.4, 10, 1.4, 17, 3.7, 6.0, 1.8, 2.4])
    # N_H_hi_4U = np.array([15, 3.8, 14, 3.6, 11, 1.7, 16, 3.7, 15.4, 2.4, 2.2])
    #
    # x_4U = np.array([5.25, 4.02, 5.40, 3.90, 4.95, 3.39, 5.21, 3.86, 5.41, 4.51, 4.38])
    # x_err_lo_4U = np.array([0.16, 0.10, 0.19, 0.07, 0.10, 0.07, 0.12, 0.06, 0.27, 0.07, 0.06])
    # x_err_hi_4U = np.array([0.12, 0.10, 0.11, 0.07, 0.10, 0.09, 0.14, 0.06, 0.18, 0.17, 0.06])

    # Zone 1
    N_H_4U_z1 = np.array([54, 43.5, 38.4, 56, 23.7, 3.5,])
    N_H_lo_4U_z1 = np.array([17, 17, 10, 17, 6.0, 1.8,])
    N_H_hi_4U_z1 = np.array([15, 14, 11, 16, 15.4, 2.4,])

    x_4U_z1 = np.array([5.25, 5.40, 4.95, 5.21, 5.41, 4.51,])
    x_err_lo_4U_z1 = np.array([0.16, 0.19, 0.10, 0.12,  0.07, 0.06])
    x_err_hi_4U_z1 = np.array([0.12, 0.11, 0.10, 0.14, 0.17, 0.06])

    # Zone 2
    N_H_4U_z2 = np.array([14.4, 17.6, 12.4, 21.3, 7.7])
    N_H_lo_4U_z2 = np.array([3.0, 3.4, 1.4, 3.7, 2.4])
    N_H_hi_4U_z2 = np.array([3.8, 3.6, 1.7, 3.7, 2.2])

    x_4U_z2 = np.array([4.02, 3.90, 3.39, 3.86, 4.38])
    x_err_lo_4U_z2 = np.array([0.10, 0.07, 0.07, 0.06,0.06])
    x_err_hi_4U_z2 = np.array([0.10, 0.07, 0.09, 0.06, 0.06])

    # 4U 1630-472 (other source)
    # first four paired observations
    plot_nh_xi(
        ax,
        N_H_4U_z1[:-2], N_H_lo_4U_z1[:-2], N_H_hi_4U_z1[:-2],
        x_4U_z1[:-2], x_err_lo_4U_z1[:-2], x_err_hi_4U_z1[:-2],
        marker='s', color='red',
        NH2=N_H_4U_z2[:-1], NH_err_lo2=N_H_lo_4U_z2[:-1],
        NH_err_hi2=N_H_hi_4U_z2[:-1],
        x2=x_4U_z2[:-1], x_err_lo2=x_err_lo_4U_z2[:-1],
        x_err_hi2=x_err_hi_4U_z2[:-1],
        color2='red'
    )

    # intermediate observation
    plot_nh_xi(
        ax,
        N_H_4U_z1[-2:-1], N_H_lo_4U_z1[-2:-1], N_H_hi_4U_z1[-2:-1],
        x_4U_z1[-2:-1], x_err_lo_4U_z1[-2:-1], x_err_hi_4U_z1[-2:-1],
        marker='s', color='magenta',
        NH2=N_H_4U_z2[-1:], NH_err_lo2=N_H_lo_4U_z2[-1:],
        NH_err_hi2=N_H_hi_4U_z2[-1:],
        x2=x_4U_z2[-1:], x_err_lo2=x_err_lo_4U_z2[-1:],
        x_err_hi2=x_err_hi_4U_z2[-1:],
        color2='magenta'
    )

    # remaining zone-1 point, which has no corresponding zone-2 measurement
    plot_nh_xi(
        ax,
        N_H_4U_z1[-1:], N_H_lo_4U_z1[-1:], N_H_hi_4U_z1[-1:],
        x_4U_z1[-1:], x_err_lo_4U_z1[-1:], x_err_hi_4U_z1[-1:],
        marker='s', color='magenta'
    )

    # =====================================================================
    # Miller+20
    # =====================================================================
    N_H_GRSh = np.array([65, 28])
    N_H_lo_GRSh = np.array([2, 4])
    N_H_hi_GRSh = np.array([1, 5])

    x_GRSh = np.array([4.2, 3.0])
    x_err_lo_GRSh = np.array([0.3, 0.1])
    x_err_hi_GRSh = np.array([0.1, 0.1])

    # GRSh
    plot_nh_xi(
        ax, N_H_GRSh, N_H_lo_GRSh, N_H_hi_GRSh,
        x_GRSh, x_err_lo_GRSh, x_err_hi_GRSh,
        marker='D', color='blue', two_zones=True
    )

    # =====================================================================
    #  GX 13+1 - XRISM2025
    # =====================================================================

    # --- GX 13+1 (slow and fast absorption zones) ---
    NH_GX13p1 = np.array([132.0, 79.0])
    NH_err_lo_GX13p1 = np.array([8.0, 9.0])
    NH_err_hi_GX13p1 = np.array([7.0, 9.0])
    logxi_GX13p1 = np.array([3.88, 4.69])
    logxi_err_lo_GX13p1 = np.array([0.01, 0.04])
    logxi_err_hi_GX13p1 = np.array([0.01, 0.03])

    plot_nh_xi(
        ax, NH_GX13p1, NH_err_lo_GX13p1, NH_err_hi_GX13p1,
        logxi_GX13p1, logxi_err_lo_GX13p1, logxi_err_hi_GX13p1,
        marker='X', color='orange', two_zones=True
    )

    # =====================================================================
    # Allen+18 -- "Two warmabs Components" fits, all 5 ObsIDs
    # (this was mislabeled as Rogantini+25 in rogantini25_params.py -- fixed here)
    # N_H in units of 1e22 cm^-2. All uncertainties given as symmetric (+-).
    # =====================================================================

    ObsID_Allen18 = np.array([2708, 11815, 11816, 11814, 11817])

    # --- Component 1 ---
    NH_1_Allen18 = np.array([0.8, 0.6, 0.3, 0.5, 0.7])
    NH_err_lo_1_Allen18 = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    NH_err_hi_1_Allen18 = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    logxi_1_Allen18 = np.array([2.98, 2.92, 2.86, 2.71, 2.88])
    logxi_err_lo_1_Allen18 = np.array([0.03, 0.01, 0.10, 0.05, 0.07])
    logxi_err_hi_1_Allen18 = np.array([0.03, 0.01, 0.10, 0.05, 0.07])

    # --- Component 2 ---
    NH_2_Allen18 = np.array([22.1, 17.7, 17.8, 12.5, 13.5])
    NH_err_lo_2_Allen18 = np.array([1.8, 0.7, 1.5, 1.3, 1.3])
    NH_err_hi_2_Allen18 = np.array([1.8, 0.7, 1.5, 1.3, 1.3])

    logxi_2_Allen18 = np.array([4.10, 3.99, 4.21, 4.07, 4.05])
    logxi_err_lo_2_Allen18 = np.array([0.04, 0.02, 0.05, 0.05, 0.05])
    logxi_err_hi_2_Allen18 = np.array([0.04, 0.02, 0.05, 0.05, 0.05])

    # =====================================================================
    # Rogantini+25 -- "Ionized disk wind" fits, pionW1 and pionW2 components.
    # N_H given in the table as 1e23 cm^-2; converted here to 1e22 cm^-2
    # (i.e. multiplied by 10) so it's consistent with plot_nh_xi's y = log10(NH)+22.
    # =====================================================================

    # --- pionW1 ---
    NH_W1_Rogantini25 = np.array([18.0, 20.0, 6.0, 53.0])
    NH_err_lo_W1_Rogantini25 = np.array([6.0, 11.0, 2.0, 12.0])
    NH_err_hi_W1_Rogantini25 = np.array([10.0, 15.0, 5.0, 12.0])

    logxi_W1_Rogantini25 = np.array([4.0, 3.8, 3.8, 3.83])
    logxi_err_lo_W1_Rogantini25 = np.array([0.1, 0.2, 0.2, 0.05])
    logxi_err_hi_W1_Rogantini25 = np.array([0.1, 0.1, 0.2, 0.05])

    # --- pionW2 ---
    NH_W2_Rogantini25 = np.array([48.0, 58.0, 99.0, 37.0])
    NH_err_lo_W2_Rogantini25 = np.array([27.0, 16.0, 39.0, 10.0])
    NH_err_hi_W2_Rogantini25 = np.array([91.0, 12.0, 25.0, 14.0])

    logxi_W2_Rogantini25 = np.array([4.7, 4.5, 4.7, 4.3])
    logxi_err_lo_W2_Rogantini25 = np.array([0.2, 0.2, 0.1, 0.1])
    logxi_err_hi_W2_Rogantini25 = np.array([0.3, 0.7, 0.1, 0.1])

    # =====================================================================
    # Plotting
    # =====================================================================

    # Allen+18, component 1 and 2
    # everything seems to be intermediate here
    plot_nh_xi(
        ax,
        NH_1_Allen18, NH_err_lo_1_Allen18, NH_err_hi_1_Allen18,
        logxi_1_Allen18, logxi_err_lo_1_Allen18, logxi_err_hi_1_Allen18,
        marker='X', color='magenta',
        NH2=NH_2_Allen18,
        NH_err_lo2=NH_err_lo_2_Allen18,
        NH_err_hi2=NH_err_hi_2_Allen18,
        x2=logxi_2_Allen18,
        x_err_lo2=logxi_err_lo_2_Allen18,
        x_err_hi2=logxi_err_hi_2_Allen18,
        color2='magenta'
    )

    # Rogantini+25, pionW1 and pionW2
    # these obs are soft in X-rays even if there is sometimes a radio flare
    plot_nh_xi(
        ax,
        NH_W1_Rogantini25[[0, 2, 3]],
        NH_err_lo_W1_Rogantini25[[0, 2, 3]],
        NH_err_hi_W1_Rogantini25[[0, 2, 3]],
        logxi_W1_Rogantini25[[0, 2, 3]],
        logxi_err_lo_W1_Rogantini25[[0, 2, 3]],
        logxi_err_hi_W1_Rogantini25[[0, 2, 3]],
        marker='X', color='red',
        NH2=NH_W2_Rogantini25[[0, 2, 3]],
        NH_err_lo2=NH_err_lo_W2_Rogantini25[[0, 2, 3]],
        NH_err_hi2=NH_err_hi_W2_Rogantini25[[0, 2, 3]],
        x2=logxi_W2_Rogantini25[[0, 2, 3]],
        x_err_lo2=logxi_err_lo_W2_Rogantini25[[0, 2, 3]],
        x_err_hi2=logxi_err_hi_W2_Rogantini25[[0, 2, 3]],
        color2='red'
    )

    # this one is intermediate
    plot_nh_xi(
        ax,
        NH_W1_Rogantini25[[1]],
        NH_err_lo_W1_Rogantini25[[1]],
        NH_err_hi_W1_Rogantini25[[1]],
        logxi_W1_Rogantini25[[1]],
        logxi_err_lo_W1_Rogantini25[[1]],
        logxi_err_hi_W1_Rogantini25[[1]],
        marker='X', color='magenta',
        NH2=NH_W2_Rogantini25[[1]],
        NH_err_lo2=NH_err_lo_W2_Rogantini25[[1]],
        NH_err_hi2=NH_err_hi_W2_Rogantini25[[1]],
        x2=logxi_W2_Rogantini25[[1]],
        x_err_lo2=logxi_err_lo_W2_Rogantini25[[1]],
        x_err_hi2=logxi_err_hi_W2_Rogantini25[[1]],
        color2='magenta'
    )

    plt.xlim(0, 6.05)
    plt.ylim(18,24.5)

    ax_obj = plt.twinx()
    ax_obj.set_xlim(ax.get_xlim())
    ax_obj.set_ylim(ax.get_ylim())

    # ax_obj.yaxis.set_visible(False)

    ax_obj.scatter([], [], marker='s', label='4U 1630-47', color='black')
    ax_obj.scatter([], [], marker='o', label='GRO J1655-40', color='black')
    ax_obj.scatter([], [], marker='D', label='GRS 1915+105', color='black')
    ax_obj.scatter([], [], marker='X', label='GX 13+1', color='black')
    ax_obj.scatter([], [], marker='<', label='H 1743-122', color='black')

    ax_obj.yaxis.set_visible(False)

    lim_NA_1e10_5ks_2sigma = np.array([[0., 1.163038007000000013e-01],
                                       [0.5, 1.148900907999999971e-01],
                                       [1.0, 4.938150109000000176e-02],
                                       [1.5, 3.152955156000000064e-02],
                                       [2., 2.437411701999999888e-02],
                                       [2.5, 5.248109659999999899e-02],
                                       [3, 1.393062539000000100e-01],
                                       [4, 7.465951670999999568e-01]
                                       ]).T

    lim_NA_1e9_5ks_3sigma = np.array([[4.0, 2.844292897999999847e-01],
                                       [4.5, 8.620929412000000180e-01],
                                       # lower limit
                                       [5.0, 3.134139536000000170e+00]
                                       ]).T

    lim_NA_1e10_5ks_3sigma = np.array([[0., 2.039911264000000002e-01],
                                       [0.5, 1.571857617000000096e-01],
                                       [1.0, 7.572973450000000661e-02],
                                       [1.5, 4.782817921999999e-02],
                                       [2., 3.723799934000000117e-02],
                                       [2.5, 7.786750087999999570e-02],
                                       [3, 2.020194187999999891e-01],
                                       [3.5, 5.130234104999999989e-01],
                                       [4.0, 1.047411674999999986e+00],
                                       [4.5, 3.262593123999999900e+00],
                                       # lower limit
                                       [5.0, 9.977716623999999257e+00]
                                       ]).T


    lim_NA_1e11_5ks_3sigma = np.array([[0., 6.476325112999999911e-01],
                                       [0.5, 5.651231459000000124e-01],
                                       [1.0, 2.796046930000000152e-01],
                                       [1.5, 2.032934363999999994e-01],
                                       [2., 2.328469434000000060e-01],
                                       [2.5, 3.717595148999999766e-01],
                                       [3, 9.976378074999999734e-01],
                                       [3.5, 3.484078982999999852e+00],
                                       [4.0, 9.987296364000000537e+00],
                                       # lower limit

                                       [4.5, 1.000000000000000000e+01],
                                       # lower limit
                                       [5.0, 1.000000000000000000e+01]
                                       ]).T

    # flux_scalings (need to be remade)
    slope_NA_3sigma = np.array([[0., -0.52],
                                [0.5, -0.56],
                                [1.0, -0.57],
                                [1.5, -0.63],
                                [2., -0.80],
                                [2.5, -0.68],
                                [3, -0.69],
                                #should be recalculated
                                [3.5, -0.55],
                                [4.0, -0.55],
                                [4.5, -0.58],
                                # lower limit
                                [5.0, -0.58]
                                ]).T

    # note: base flux for 0p1 Edd is 7.824e-09
    c_rescale = np.log10(flux_lim_NA / 1e-10)

    resc_lim_NA = np.log10(lim_NA_1e10_5ks_3sigma[1]) + c_rescale * slope_NA_3sigma[1]

    ax_obj.plot(lim_NA_1e10_5ks_3sigma[0], resc_lim_NA + 22, ls='--', color='grey',
                label=r'NewAthena'+'\n'+r'detectability')
    plt.legend(loc='lower right')

    plt.tight_layout()