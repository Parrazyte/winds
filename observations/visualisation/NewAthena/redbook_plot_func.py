import os
import matplotlib.pyplot as plt
from xspec import *
from xspec_config_multisp import set_ener, rebinv_xrism, xPlot
import numpy as np
from matplotlib.table import Table
from matplotlib.patches import FancyBboxPatch

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

def plot_redbook_lowE(figsize=(10,8)):

    os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/NewAthena/SpecialIssue/Redbook/SIXTE/common')
    Xset.restore('mod_full_NA_Chandra.xcm')

    AllData.ignore('**-0.8')

    set_ener('large_canon',xrism=True)

    rebinv_xrism(1,10,max_bins=60000)
    rebinv_xrism(2,10,max_bins=60000)
    rebinv_xrism(3,10,max_bins=60000)
    rebinv_xrism(4,10,max_bins=60000)
    rebinv_xrism(5,10,max_bins=60000)
    rebinv_xrism(6,10,max_bins=60000)
    rebinv_xrism(7,5,max_bins=60000)
    rebinv_xrism(8,5,max_bins=60000)
    rebinv_xrism(9,5,max_bins=60000)
    rebinv_xrism(10,3,max_bins=60000)
    rebinv_xrism(11,3,max_bins=60000)
    rebinv_xrism(12,3,max_bins=60000)

    Plot.add=False
    n=15
    xPlot('eeuf',xlims=[0.8,2.05],mult_factors=np.array([n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0,
                                                         n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0]),
          data_colors=['orange','darkred','darkblue','orange','darkred','darkblue',
                       'grey','grey','grey','grey','grey','grey'],
          group_names=[r"log$\xi$=4($\pm0.01$) | log$_{10}$NH=23($\pm0.01$) cm$^{-2}$""\n"
                       +r"v$_{turb}$=100($\pm3$) km s$^{-1}$ | v$_{out}$=300($\pm3$) km s$^{-1}$",
                       r"log$\xi$=2($\pm0.03$) | log$_{10}$NH=22($\pm0.004$) cm$^{-2}$""\n"
                       +r"v$_{turb}$=100($\pm1$) km s$^{-1}$ | v$_{out}$=300($\pm2$) km s$^{-1}$",
                      "",
                       "","","",
                       "","","",
                       "","",""],
          data_alpha=[1, 1, 1, 1, 1,1, 0.5, 0.5, 0.5, 0.5,0.5,0.5],
          model_colors=['orange','darkred','darkblue','orange','darkred','darkblue',
                        'None','None','None','None','None','None'],
          model_ls=['-', '-', '-', ':', ':', ':',
                    '-', '-', '-', ':', ':', ':', ],
          ylims=np.array([1e-9,2.5]),auto_figsize=figsize)
    plt.yscale('log')
    ax=plt.gca()
    # ax.legend(loc='lower right')
    plt.tight_layout()
    rebinv_xrism(2,5,max_bins=60000)
    rebinv_xrism(8,5,max_bins=60000)

    ax.set_ylabel(r'keV$^{2}$ (Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$) $-$ shifted for clarity $-$')

    #axins.remove()
    axins = ax.inset_axes(
        [0.555, 0.375, 0.25+(0.03 if figsize[0]<9 else 0), 0.215],
        xlim=(1.99, 2.01), ylim=(1e-2, 1e0), xticklabels=[], yticklabels=[])
    xPlot('eeuf',axes_input=[axins],mult_factors=[n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0,
                                                  n**-5.4,n**-2.35,n**-0.5,n**-5.1,n**-1.5,n**-0],
          data_colors=['orange','darkred','darkblue','orange','darkred','darkblue',
                       'grey','grey','grey','grey','grey','grey'],
          model_colors=['orange','darkred','darkblue','orange','darkred','darkblue',
                        'None','None','None','None','None','None'],
          model_ls=['-','-','-','--','--','--',
                        '-','-','-','--','--','--',],
          data_alpha=[1, 1, 1, 1, 1,1, 0.5, 0.5, 0.5, 0.5,0.5,0.5],)
    axins.set_ylabel('')
    axins.set_xlabel('')
    axins.set_title('')
    if figsize[0]>9:
        axins.set_xlim(1.9975,2.0125)

    else:
        axins.set_xlim(1.9975,2.0105)

    axins.set_ylim(2.5e-4,3e-3)
    axins.get_children()[-3].remove()
    axins.get_children()[-2].remove()
    ax.indicate_inset_zoom(axins, edgecolor="black",alpha=0.5)
    axins.set_yscale('log')

    axins.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True,
        labeltop=False,
        direction='in')
    axins.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=True,  # ticks along the top edge are off
        direction='in',labelleft=False,labelright=figsize[0]>=9)

    if figsize[0]>9:
        axins.set_xticklabels([1.995,2.000,2.005,2.010])
    else:
        axins.set_xticklabels(['',1.995,'',2.01])

    axins.text(1.998, 6.6e-4, r'v=300$\pm$9$\,$km$\,$s$^{-1}$', fontsize=8 if figsize[0]<9 else 10)
    axins.text(1.998, 4.5e-4, r'$\sigma$=100$\pm$12$\,$km$\,$s$^{-1}$', fontsize=8 if figsize[0]<9 else 10)
    axins.text(1.998, 3.2e-4, r'EW=3.18$\pm$0.05$\,$eV', fontsize=8 if figsize[0]<9 else 10)

    if figsize[0]<9:
        axins.tick_params(axis='x', labelsize=8)
        axins.tick_params(axis='y', labelsize=8)

    # ax.legend(loc='lower right')
    plt.tight_layout()
    ax.get_children()[-2].remove()
    # ax2 = ax.twinx()
    # ax2.errorbar([],[],xerr=[],yerr=[],color='darkblue',
    #              label=r"log$\xi$=4($\pm0.2$) | log$_{10}$NH=23($\pm0.2$) cm$^{-2}$""\n"
    #                    +r"v$_{turb}$=100($\pm30$) km s$^{-1}$ | v$_{out}$=300($\pm60$) km s$^{-1}$")
    # ax2.errorbar([],[],xerr=[],yerr=[],color='orange',
    #              label=r"log$\xi$=2($\pm0.1$) | log$_{10}$NH=22($\pm0.1$) cm$^{-2}$""\n"
    #                    +r"v$_{turb}$=100($\pm15$) km s$^{-1}$ | v$_{out}$=300($\pm30$) km s$^{-1}$")
    # ax2.get_yaxis().set_visible(False)
    # ax2.legend(loc='lower right')

    '''
    invert blue yellow and red to get the logxi=0 above - it will make it easier to plot the high energy panel
    perhaps 6.5-8.5 with zoom or 6.5-7.0 - no need to put the 50s when they cannot do anything
    '''

    # ---------------------------------------------------------
    # Colors
    # ---------------------------------------------------------

    # c1 = "darkblue"
    # c2 = "orange"
    # c3 = "darkred"

    #to match the graph
    c3 = "darkblue"
    c1 = "orange"
    c2 = "darkred"


    # ---------------------------------------------------------
    # Legend contents
    # ---------------------------------------------------------

    data = [
        [
            r"log$\xi$ =",
            r"4$\pm0.02$",
            r"2$\pm0.003$",
            r"0$\pm0.01$",
        ],
        [
            r"log$_{10}$NH [cm$^{-2}$] =",
            r"23$\pm0.03$",
            r"22$\pm0.001$",
            r"22$\pm0.01$",
        ],
        [
            r"$v_{\rm turb} $ [km s$^{-1}$]=",
            r"100$\pm6$",
            r"100$\pm2$",
            r"100$\pm16$",
        ],
        [
            r"$v_{\rm out}$ [km s$^{-1}$] =",
            r"300$\pm5$",
            r"300$\pm2$",
            r"300$\pm40$",
        ],
    ]

    colors = [
        ["black", c1, c2, c3],
        ["black", c1, c2, c3],
        ["black", c1, c2, c3],
        ["black", c1, c2, c3],
    ]


    # ---------------------------------------------------------
    # Position and size of the legend
    # ---------------------------------------------------------
    #
    # All values are in axes coordinates:
    # [left, bottom, width, height]
    #

    bbox = [0.50, 0.02, 0.48, 0.19]


    # ---------------------------------------------------------
    # Draw the legend border FIRST
    # ---------------------------------------------------------

    border = FancyBboxPatch(
        (bbox[0], bbox[1]),
        bbox[2],
        bbox[3],
        boxstyle="round,pad=0.008,rounding_size=0.008",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="grey",
        linewidth=0.8,
        zorder=2,
    )

    ax.add_patch(border)

    ax.text(
        bbox[0] + 0.35+(0.065 if figsize[0]<9 else 0),
        bbox[1] + bbox[3] - (0.008 if figsize[0]>=9 else 0.006),

        "5ks NewAthena simulations",
        transform=ax.transAxes,
        color="black",
        fontsize=10 if figsize[0]<9 else 11,
        ha="right",
        va="top",
        zorder=3,
    )

    # ---------------------------------------------------------
    # Create the table
    # ---------------------------------------------------------

    table_bbox = [
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3] - 0.035,
    ]
    table = Table(
        ax,
        bbox=table_bbox,
    )


    # ---------------------------------------------------------
    # Column widths
    # ---------------------------------------------------------
    #
    # These are relative widths. Reducing them reduces the
    # horizontal space occupied by the legend.
    #
    # The first column is wider because it contains the labels
    # and units. The three value columns are deliberately narrow.
    #

    col_widths = [
        0.35,   # labels
        0.23,   # dark blue
        0.23,   # orange
        0.21,   # dark red
    ]


    # Row height
    row_height = 0.11


    # ---------------------------------------------------------
    # Add cells
    # ---------------------------------------------------------

    for i, row in enumerate(data):

        for j, value in enumerate(row):

            cell = table.add_cell(
                i,
                j,
                width=col_widths[j],
                height=row_height,
                text=value,
                loc="left",
                facecolor="white",
                edgecolor="none",
            )

            cell.get_text().set_color(colors[i][j])
            cell.get_text().set_fontsize(11)

            # Reduce padding inside each cell.
            cell.PAD = 0.0


    # ---------------------------------------------------------
    # Make all cell borders invisible
    # ---------------------------------------------------------

    for cell in table.get_celld().values():
        cell.set_linewidth(0)
        cell.set_edgecolor("none")


    # ---------------------------------------------------------
    # Add table on top of the border
    # ---------------------------------------------------------

    table.set_zorder(3)
    ax.add_table(table)
    plt.show()

    ax_secleg=plt.twinx()
    ax_secleg.axis('off')


    class TriColorRect:
        def __init__(self, colors=("red", "yellow", "blue")):
            self.colors = colors

    class HandlerTriColorRect(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            x0, y0 = -xdescent, -ydescent
            n = len(orig_handle.colors)
            w = width / n
            artists = [
                Rectangle((x0 + i * w, y0), w, height,
                          facecolor=c, edgecolor="none", transform=trans)
                for i, c in enumerate(orig_handle.colors)
            ]
            artists.append(Rectangle((x0, y0), width, height,
                                     fill=False, edgecolor="k", lw=0.5,
                                     transform=trans))
            return artists

    class ErrBarProxy:
        """Cross-shaped errorbar. Sizes are fractions of the handle box (0-1)."""

        def __init__(self, x_frac, y_frac):
            self.x_frac = x_frac  # total horizontal extent / handle width
            self.y_frac = y_frac  # total vertical extent / handle height

    class HandlerErrBar(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            xc = -xdescent + width / 2
            yc = -ydescent + height / 2
            hx = orig_handle.x_frac * width / 2
            hy = orig_handle.y_frac * height / 2
            kw = dict(color="k", lw=1, transform=trans)

            return [
                Line2D([xc - hx, xc + hx], [yc, yc], **kw),
                Line2D([xc, xc], [yc - hy, yc + hy], **kw),
            ]
    # # --- build the legend on ax_secleg ---
    # fig, (ax, ax_secleg) = plt.subplots(1, 2, figsize=(8, 4))
    # ax_secleg.axis("off")

    handles = [
        TriColorRect(("darkblue", "darkred","orange")),
        Rectangle((0, 0), 1, 1., facecolor="grey", edgecolor="k", lw=0.5),
        # ErrBarProxy(x_frac=0.4, y_frac=0.4),  # small
        # ErrBarProxy(x_frac=1.5, y_frac=1.0),  # large
        Line2D([], [], color="k", ls="-", lw=1),   # plain line
        Line2D([], [], color="k", ls=":", lw=1),
    ]
    labels = ["NewAthena", "Chandra", "5ks", "50s"]

    ax_secleg.legend(
        handles, labels,
        handler_map={
            TriColorRect: HandlerTriColorRect(),
            ErrBarProxy: HandlerErrBar(),
        },
        loc="upper left", frameon=True,
        handlelength=2.0, handleheight=1.5, labelspacing=1.0,ncol=2,columnspacing=0.5,
        fontsize=9 if figsize[0]<9 else None,
    )

    plt.tight_layout()

    # plt.show()
'''
HIGH ENERGIES 
'''

def plot_redbook_highE(figsize=(10,8)):

    os.chdir('/media/parrazyte/crucial_SSD/Observ/highres/NewAthena/SpecialIssue/Redbook/SIXTE/common')
    Xset.restore('mod_full_NA_Chandra.xcm')
    set_ener('large_canon',xrism=True)

    n=15

    set_ener('large_canon',xrism=True)

    rebinv_xrism(1,10,max_bins=60000)
    rebinv_xrism(2,10,max_bins=60000)
    rebinv_xrism(3,10,max_bins=60000)
    rebinv_xrism(4,10,max_bins=60000)
    rebinv_xrism(5,10,max_bins=60000)
    rebinv_xrism(6,10,max_bins=60000)
    rebinv_xrism(7,5,max_bins=60000)
    rebinv_xrism(8,5,max_bins=60000)
    rebinv_xrism(9,5,max_bins=60000)
    rebinv_xrism(10,3,max_bins=60000)
    rebinv_xrism(11,3,max_bins=60000)
    rebinv_xrism(12,3,max_bins=60000)

    xPlot('eeuf',xlims=[6.6,8.55],mult_factors=np.array([n**-2.5,n**-1.5,n**-0.5,n**-2.,n**-1.,n**-0,
                                                         n**-2.5,n**-1.5,n**-0.5,n**-2.,n**-1.,n**-0]),
          data_colors=['orange','darkred','darkblue', 'orange','darkred','darkblue',
                       'grey', 'grey', 'grey', 'grey', 'grey', 'grey'],
          group_names=[r"log$\xi$=4($\pm0.01$) | log$_{10}$NH=23($\pm0.01$) cm$^{-2}$""\n"
                       + r"v$_{turb}$=100($\pm3$) km s$^{-1}$ | v$_{out}$=300($\pm3$) km s$^{-1}$",
                       r"log$\xi$=2($\pm0.03$) | log$_{10}$NH=22($\pm0.004$) cm$^{-2}$""\n"
                       + r"v$_{turb}$=100($\pm1$) km s$^{-1}$ | v$_{out}$=300($\pm2$) km s$^{-1}$",
                       "",
                       "", "", "",
                       "", "", "",
                       "", "", ""],
          data_alpha=[1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
          model_colors=['orange','darkred','darkblue','orange','darkred','darkblue',
                        'None', 'None', 'None', 'None', 'None', 'None'],
          model_ls=['-', '-', '-', ':', ':', ':',
                    '-', '-', '-', ':', ':', ':', ],
          ylims=np.array([7e-7,5]),auto_figsize=figsize)
    ax=plt.gca()
    plt.yscale('log')
    ax.get_children()[-2].remove()
    plt.tight_layout()

    ax.set_ylabel(r'keV$^{2}$ (Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$) $-$ shifted for clarity $-$')

    rebinv_xrism(1,3,max_bins=60000)
    rebinv_xrism(7,3,max_bins=60000)
    axins = ax.inset_axes(
        [0.26, 0.055, 0.305, 0.40],
        xlim=(6.6, 8.55), ylim=(2e-6, 4), xticklabels=[], yticklabels=[])
    xPlot('eeuf',axes_input=[axins],mult_factors=np.array([n**-2.5,n**-1.5,n**-0.5,n**-2.,n**-1.,n**-0,
                                                           n**-2.5,n**-1.5,n**-0.5,n**-2.,n**-1.,n**-0]),
          data_colors=['orange','darkred','darkblue','orange','darkred','darkblue',
                       'grey','grey','grey','grey','grey','grey'],
          model_colors=['orange','darkred','darkblue','orange','darkred','darkblue',
                        'None','None','None','None','None','None'],
          model_ls=['-', '-', '-', ':', ':', ':',
                    '-', '-', '-', ':', ':', ':', ],
          data_alpha=[1, 1, 1, 1, 1,1, 0.5, 0.5, 0.5, 0.5,0.5,0.5],)

    axins.set_ylabel('')
    axins.set_xlabel('')
    axins.set_title('')
    axins.set_xlim(6.945, 6.99)
    if figsize[0]<9:
        axins.set_ylim(2e-6 * (3.2 / 4), 3.2e-3)
    else:
        axins.set_ylim(2e-6, 4e-3)
    axins.get_children()[-3].remove()
    axins.get_children()[-2].remove()
    axins.set_yscale('log')
    axins.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True,
        labeltop=False,
        direction='out')
    axins.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=True,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        direction='out',labelleft=True,labelright=False)

    if figsize[0]<9:
        axins.set_xticklabels(['',6.96,6.98,])
        # axins.set_yticklabels(['','','1e-5','','','1e-4','','','1e-3'])
        axins.tick_params(axis='y', labelsize=8)
    else:
        axins.set_xticklabels(['',6.95,6.96,6.97,6.98,])

    axins.text(6.947, 13.2e-6*0.7, r'v = 300$\pm$10 km s$^{-1}$', fontsize=9 if figsize[0]<9 else 11)
    axins.text(6.947, 6.4e-6*0.7, r'$\sigma$ = 100$\pm$17 km s$^{-1}$', fontsize=9 if figsize[0]<9 else 11)
    axins.text(6.947, 3.2e-6*0.7, r'EW$_{1/2+3/2}$ = 20.5$\pm$0.6 eV', fontsize=9 if figsize[0]<9 else 11)
    inset_ind=ax.indicate_inset_zoom(axins, edgecolor="black",alpha=0.5)
    inset_ind.connectors[0].set_visible(True)
    inset_ind.connectors[1].set_visible(True)
    inset_ind.connectors[2].set_visible(False)
    inset_ind.connectors[3].set_visible(True)
    plt.show()


    if figsize[0]>=9:





        # ---------------------------------------------------------
        # Colors
        # ---------------------------------------------------------

        # c1 = "darkblue"
        # c2 = "orange"
        # c3 = "darkred"

        #to match the graph
        c3 = "darkblue"
        c1 = "orange"
        c2 = "darkred"

        # ---------------------------------------------------------
        # Legend contents
        # ---------------------------------------------------------

        data = [
            [
                r"log$\xi$ =",
                r"4($\pm0.3$)",
                r"2($\pm0.03$)",
                r"0($\pm0.3$)",
            ],
            [
                r"log$_{10}$NH [cm$^{-2}$] =",
                r"23($\pm0.3$)",
                r"22($\pm0.1$)",
                r"22($\pm0.1$)",
            ],
            [
                r"$v_{\rm turb} $ [km s$^{-1}$]=",
                r"100(<120)",
                r"100($\pm20$)",
                r"100(<600)",
            ],
            [
                r"$v_{\rm out}$ [km s$^{-1}$] =",
                r"300($\pm50$)",
                r"300($\pm30$)",
                r"300($\pm550$)",
            ],
        ]

        colors = [
            ["black", c1, c2, c3],
            ["black", c1, c2, c3],
            ["black", c1, c2, c3],
            ["black", c1, c2, c3],
        ]


        # ---------------------------------------------------------
        # Position and size of the legend
        # ---------------------------------------------------------
        #
        # All values are in axes coordinates:
        # [left, bottom, width, height]
        #

        bbox = [0.59, 0.02, 0.4, 0.19]


        # ---------------------------------------------------------
        # Draw the legend border FIRST
        # ---------------------------------------------------------

        border = FancyBboxPatch(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            boxstyle="round,pad=0.008,rounding_size=0.008",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="grey",
            linewidth=0.8,
            zorder=2,
        )

        ax.add_patch(border)

        ax.text(
            bbox[0] + 0.31,
            bbox[1] + bbox[3] - 0.008,

            "50s NewAthena simulations",
            transform=ax.transAxes,
            color="black",
            fontsize=11,
            ha="right",
            va="top",
            zorder=3,
        )

        # ---------------------------------------------------------
        # Create the table
        # ---------------------------------------------------------

        table_bbox = [
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3] - 0.035,
        ]
        table = Table(
            ax,
            bbox=table_bbox,
        )


        # ---------------------------------------------------------
        # Column widths
        # ---------------------------------------------------------
        #
        # These are relative widths. Reducing them reduces the
        # horizontal space occupied by the legend.
        #
        # The first column is wider because it contains the labels
        # and units. The three value columns are deliberately narrow.
        #

        col_widths = [
            0.36,   # labels
            0.23,   # dark blue
            0.20,   # orange
            0.20,   # dark red
        ]


        # Row height
        row_height = 0.11


        # ---------------------------------------------------------
        # Add cells
        # ---------------------------------------------------------

        for i, row in enumerate(data):

            for j, value in enumerate(row):

                cell = table.add_cell(
                    i,
                    j,
                    width=col_widths[j],
                    height=row_height,
                    text=value,
                    loc="left",
                    facecolor="white",
                    edgecolor="none",
                )

                cell.get_text().set_color(colors[i][j])
                cell.get_text().set_fontsize(10)

                # Reduce padding inside each cell.
                cell.PAD = 0.0


        # ---------------------------------------------------------
        # Make all cell borders invisible
        # ---------------------------------------------------------

        for cell in table.get_celld().values():
            cell.set_linewidth(0)
            cell.set_edgecolor("none")


        # ---------------------------------------------------------
        # Add table on top of the border
        # ---------------------------------------------------------

        table.set_zorder(3)
        ax.add_table(table)
        plt.show()

    else:

        # ---------------------------------------------------------
        # Colors
        # ---------------------------------------------------------

        # to match the graph
        c3 = "darkblue"
        c1 = "orange"
        c2 = "darkred"

        # ---------------------------------------------------------
        # Legend contents
        # ---------------------------------------------------------
        # Header row = parameter names (previously the leftmost column).
        # Each following row = one simulation (one color).

        header = [
            r"log$\xi$",
            r"log$_{10}$N$_{\rm H}$" "\n" r"cm$^{-2}$",
            r"$v_{\rm turb}$" "\n" r"km s$^{-1}$",
            r"$v_{\rm out}$" "\n" r"km s$^{-1}$",
        ]

        rows = [
            # c1 (orange)
            [r"4$\pm0.3$",  r"23$\pm0.3$", r"100(<120)",     r"300$\pm50$"],
            # c2 (dark red)
            [r"2$\pm0.03$", r"22$\pm0.1$", r"100$\pm20$",  r"300$\pm30$"],
            # c3 (dark blue)
            [r"0$\pm0.3$",  r"22$\pm0.1$", r"100(<600)",     r"300$\pm550$"],
        ]

        data = [header] + rows

        row_colors = ["black", c1, c2, c3]   # one color per row (header is black)

        # ---------------------------------------------------------
        # Position and size of the legend
        # ---------------------------------------------------------
        # Axes coordinates: [left, bottom, width, height]
        # The transposed table is wider than tall, so you may want to
        # tweak the width/height to taste.

        bbox = [0.585, 0.02, 0.405, 0.32]

        # ---------------------------------------------------------
        # Draw the legend border FIRST
        # ---------------------------------------------------------

        border = FancyBboxPatch(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            boxstyle="round,pad=0.008,rounding_size=0.008",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="grey",
            linewidth=0.8,
            ls=':',
            zorder=2,
        )
        ax.add_patch(border)

        # Title, centered above the header row
        ax.text(
            bbox[0] + bbox[2] / 2,
            bbox[1] + bbox[3] - 0.008,
            "50s NewAthena simulations",
            transform=ax.transAxes,
            color="black",
            fontsize=11,
            ha="center",
            va="top",
            zorder=3,
        )

        # ---------------------------------------------------------
        # Create the table (sits below the title)
        # ---------------------------------------------------------

        title_space = 0.06
        table_bbox = [
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3] - title_space,
        ]
        table = Table(ax, bbox=table_bbox)
        table.auto_set_font_size(False)
        # ---------------------------------------------------------
        # Column widths / row heights (relative; scaled to the bbox)
        # ---------------------------------------------------------

        col_widths = [0.205, 0.245, 0.285, 0.265]   # logxi, NH, v_turb, v_out

        header_height = 0.22   # taller: two-line labels
        row_height = 0.11

        # ---------------------------------------------------------
        # Add cells
        # ---------------------------------------------------------

        for i, row in enumerate(data):
            for j, value in enumerate(row):
                cell = table.add_cell(
                    i,
                    j,
                    width=col_widths[j],
                    height=header_height if i == 0 else row_height,
                    text=value,
                    loc="center",
                    facecolor="white",
                    edgecolor="none",
                )
                cell.get_text().set_color(row_colors[i])
                cell.get_text().set_fontsize(9)
                cell.PAD = 0.0

        # ---------------------------------------------------------
        # Make all cell borders invisible
        # ---------------------------------------------------------

        for cell in table.get_celld().values():
            cell.set_linewidth(0)
            cell.set_edgecolor("none")

        # ---------------------------------------------------------
        # Add table on top of the border
        # ---------------------------------------------------------

        table.set_zorder(3)
        ax.add_table(table)
        plt.show()

    if figsize[0]>=9:
        ax_secleg=plt.twinx()
        ax_secleg.axis('off')



        class TriColorRect:
            def __init__(self, colors=("red", "yellow", "blue")):
                self.colors = colors

        class HandlerTriColorRect(HandlerBase):
            def create_artists(self, legend, orig_handle, xdescent, ydescent,
                               width, height, fontsize, trans):
                x0, y0 = -xdescent, -ydescent
                n = len(orig_handle.colors)
                w = width / n
                artists = [
                    Rectangle((x0 + i * w, y0), w, height,
                              facecolor=c, edgecolor="none", transform=trans)
                    for i, c in enumerate(orig_handle.colors)
                ]
                artists.append(Rectangle((x0, y0), width, height,
                                         fill=False, edgecolor="k", lw=0.5,
                                         transform=trans))
                return artists

        class ErrBarProxy:
            """Cross-shaped errorbar. Sizes are fractions of the handle box (0-1)."""

            def __init__(self, x_frac, y_frac):
                self.x_frac = x_frac  # total horizontal extent / handle width
                self.y_frac = y_frac  # total vertical extent / handle height

        class HandlerErrBar(HandlerBase):
            def create_artists(self, legend, orig_handle, xdescent, ydescent,
                               width, height, fontsize, trans):
                xc = -xdescent + width / 2
                yc = -ydescent + height / 2
                hx = orig_handle.x_frac * width / 2
                hy = orig_handle.y_frac * height / 2
                kw = dict(color="k", lw=1, transform=trans)

                return [
                    Line2D([xc - hx, xc + hx], [yc, yc], **kw),
                    Line2D([xc, xc], [yc - hy, yc + hy], **kw),
                ]
        # # --- build the legend on ax_secleg ---
        # fig, (ax, ax_secleg) = plt.subplots(1, 2, figsize=(8, 4))
        # ax_secleg.axis("off")

        handles = [
            TriColorRect(("darkblue", "darkred","orange")),
            Rectangle((0, 0), 1, 1., facecolor="grey", edgecolor="k", lw=0.5),
            # ErrBarProxy(x_frac=0.4, y_frac=0.4),  # small
            # ErrBarProxy(x_frac=1.5, y_frac=1.0),  # large
            Line2D([], [], color="k", ls="-", lw=1),   # plain line
            Line2D([], [], color="k", ls=":", lw=1),
        ]
        labels = ["NewAthena", "Chandra", "5ks", "50s"]

        ax_secleg.legend(
            handles, labels,
            handler_map={
                TriColorRect: HandlerTriColorRect(),
                ErrBarProxy: HandlerErrBar(),
            },
            loc="lower left", frameon=True,
            handlelength=2.0, handleheight=1.5, labelspacing=1.0,ncol=1,columnspacing=0.5,
        )

    plt.tight_layout()
    plt.subplots_adjust(right=0.995)
    plt.show()
