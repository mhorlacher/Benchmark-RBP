# /usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def fig2_layout():
    fig = plt.figure(figsize=(8.27, 11.69))

    gs = GridSpec(10, 6, figure=fig)

    ax1 = fig.add_subplot(gs[0:2, :3])
    ax2 = fig.add_subplot(gs[0:2, 3:])
    ax3 = fig.add_subplot(gs[2:5, :2])
    ax4 = fig.add_subplot(gs[2:5, 2:4])
    ax5 = fig.add_subplot(gs[2:5, 4:])

    ax6 = fig.add_subplot(gs[5:, 0:])

    ax3.set_aspect("equal")
    ax4.set_aspect("equal")
    ax5.set_aspect("equal")

    # Add the subplots
    # ax1.imshow(image1)
    # ax2.imshow(image2)
    # ax3.imshow(image3)
    # ax4.imshow(image4)

    # Adjust the size of subplots
    # fig.set_size_inches(10,8)
    plt.tight_layout()

    # Now get the panel sizes

    PANEL_SIZES = {}
    panels = [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]

    fig_size = fig.get_size_inches()

    for panel, ax in zip(panels, [ax1, ax2, ax3, ax4, ax5]):
        ax_pos = ax.get_position()
        ax_size = ax_pos.width, ax_pos.height
        ax_size_inches = ax_size[0] * fig_size[0], ax_size[1] * fig_size[1]

        PANEL_SIZES[panel] = ax_size_inches

    plt.close()
    return PANEL_SIZES


FIG2_PANEL_SIZES = fig2_layout()


def fig3_layout():
    fig = plt.figure(figsize=(8.27, 11.69))

    gs = GridSpec(5, 12, figure=fig)

    # First row
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])
    ax3 = fig.add_subplot(gs[0, 6:9])
    ax4 = fig.add_subplot(gs[0, 9:12])

    # Second row
    ax5 = fig.add_subplot(gs[1, 0:4])
    ax6 = fig.add_subplot(gs[1, 4:8])
    ax7 = fig.add_subplot(gs[1, 8:12])

    # Third row
    ax8 = fig.add_subplot(gs[2, 0:6])
    ax9 = fig.add_subplot(gs[2, 6:12])

    # Comments
    ax10 = fig.add_subplot(gs[3:, :])

    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    ax6.set_aspect("equal")
    ax7.set_aspect("equal")
    ax8.set_aspect("equal")
    ax9.set_aspect("equal")

    # Add the subplots
    # ax1.imshow(image1)
    # ax2.imshow(image2)
    # ax3.imshow(image3)
    # ax4.imshow(image4)

    # Adjust the size of subplots
    # fig.set_size_inches(10,8)
    plt.tight_layout()

    # Now get the panel sizes

    PANEL_SIZES = {}
    panels = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    fig_size = fig.get_size_inches()

    for panel, ax in zip(panels, [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]):
        ax_pos = ax.get_position()
        ax_size = ax_pos.width, ax_pos.height
        ax_size_inches = ax_size[0] * fig_size[0], ax_size[1] * fig_size[1]

        PANEL_SIZES[panel] = ax_size_inches

    plt.close()
    return PANEL_SIZES


FIG3_PANEL_SIZES = fig3_layout()
