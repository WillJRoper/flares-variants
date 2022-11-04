import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, TwoSlopeNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as eagle_io
from flare import plt as flareplt
from unyt import mh, cm, Gyr, g, Msun, Mpc
from utils import mkdir, plot_meidan_stat, get_nonmaster_evo_data
from utils import get_nonmaster_centred_data, grav_enclosed, calc_ages
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck18 as cosmo, z_at_value
from scipy.spatial import cKDTree
import cmasher as cmr


# Turn on grid
mpl.rcParams.update({"axes.grid": True})


def plot_stellarmet_mass_relation_vary(snap):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Define the path
    ini_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["flares_00", "FLARES_00_REF",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh",
             "flares_00_no_agn", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "flares_00_H_reion_z03",
             "flares_00_H_reion_z075", "flares_00_H_reion_z14"]

    # Define labels for each
    labels = ["AGNdT9", "REF", "SKIP",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{r, 0}$", "$z_{r, 7.5}$",
              "$z_{r, 14}$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]

    # Define plot dimensions
    nrows = 4
    ncols = 3

    # Define norm
    norm = LogNorm(vmin=1, vmax=10)

    # Define hexbin extent
    extent = [8, 11.5, -4, -1]

    # Set up the plots
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[-1])

    for i in range(nrows):
        for j in range(ncols):

            if i * ncols + j >= len(labels):
                continue

            if labels[i * ncols + j] == "SKIP":
                continue

            # Create axis
            ax = fig.add_subplot(gs[i, j])
            ax.grid(True)

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$Z_\star$")
            if i == nrows - 1:
                ax.set_xlabel(r"$M_\star / M_\odot$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)

            if i < nrows - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(10**extent[2], 10**extent[3])
            ax.set_xlim(10**extent[0], 10**extent[1])

            # Label axis
            ax.text(0.95, 0.1, labels[i * ncols + j],
                    bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                              alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axes.append(ax)

    for (ind, t), l in zip(enumerate(types), labels):

        path = ini_path.replace("<type>", t)

        print(path)

        mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/ApertureMeasurements/Mass/030kpc",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 10
        part_mets = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                        snap,
                                        "PartType4/SmoothedMetallicity",
                                        noH=True, physicalUnits=True,
                                        numThreads=8)
        cops = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/CentreOfPotential",
                                   noH=True, physicalUnits=True,
                                   numThreads=8) * 1000
        grps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/GroupNumber",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)
        subgrps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                      snap,
                                      "Subhalo/SubGroupNumber",
                                      noH=True, physicalUnits=True,
                                      numThreads=8)
        ini_mass = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                       snap,
                                       "PartType4/InitialMass",
                                       noH=True, physicalUnits=True,
                                       numThreads=8) * 10 ** 10
        coords = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                     snap,
                                     "PartType4/Coordinates",
                                     noH=True, physicalUnits=True,
                                     numThreads=8) * 1000
        part_grps = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                        snap,
                                        "PartType4/GroupNumber",
                                        noH=True, physicalUnits=True,
                                        numThreads=8)
        part_subgrps = eagle_io.read_array("PARTDATA",
                                           path.replace("<type>", t),
                                           snap,
                                           "PartType4/SubGroupNumber",
                                           noH=True, physicalUnits=True,
                                           numThreads=8)

        # Apply some cuts
        mokinds = mass > 10**8
        mass = mass[mokinds]
        cops = cops[mokinds, :]
        grps = grps[mokinds]
        subgrps = subgrps[mokinds]

        # Set up array to store sfrs
        mets = []
        ms = []

        # Loop over galaxies
        for igal in range(mass.size):

            # Get galaxy data
            m = mass[igal]
            cop = cops[igal, :]
            g = grps[igal]
            sg = subgrps[igal]

            # Get this galaxies stars
            sokinds = np.logical_and(part_grps == g, part_subgrps == sg)
            this_coords = coords[sokinds, :] - cop
            this_ini_mass = ini_mass[sokinds]
            this_part_mets = part_mets[sokinds]

            # Compute stellar radii
            rs = np.sqrt(this_coords[:, 0] ** 2
                         + this_coords[:, 1] ** 2
                         + this_coords[:, 2] ** 2)

            # Get only particles within the aperture
            rokinds = rs < 30
            this_ini_mass = this_ini_mass[rokinds]
            this_part_mets = this_part_mets[rokinds]

            if len(this_ini_mass) == 0:
                continue

            # Store values for plotting
            ms.append(m)
            mets.append(np.average(this_part_mets, weights=this_ini_mass))

        # Convert to arrays and get mask
        mets = np.array(mets)
        ms = np.array(ms)
        okinds = mets > 0

        im = axes[ind].hexbin(ms[okinds], mets[okinds], mincnt=1, gridsize=50,
                              xscale="log", yscale="log", linewidth=0.2,
                              cmap="plasma", norm=norm, extent=extent)

    # Set up colorbar
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$N$")

    # Save figure
    mkdir("plots/massmet/")
    fig.savefig("plots/massmet/mass_stellar_metallicity_%s.png" % snap,
                bbox_inches="tight")


def plot_gasmet_mass_relation_vary(snap):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Define the path
    ini_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["flares_00", "FLARES_00_REF",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh",
             "flares_00_no_agn", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "flares_00_H_reion_z03",
             "flares_00_H_reion_z075", "flares_00_H_reion_z14"]

    # Define labels for each
    labels = ["AGNdT9", "REF", "SKIP",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{r, 0}$", "$z_{r, 7.5}$",
              "$z_{r, 14}$"]
    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]

    # Define plot dimensions
    nrows = 4
    ncols = 3

    # Define norm
    norm = LogNorm(vmin=1, vmax=10)

    # Define hexbin extent
    extent = [8, 11.5, -4, -1]

    # Set up the plots
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[-1])

    for i in range(nrows):
        for j in range(ncols):

            if i * ncols + j >= len(labels):
                continue

            if labels[i * ncols + j] == "SKIP":
                continue

            # Create axis
            ax = fig.add_subplot(gs[i, j])
            ax.grid(True)

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$Z_\mathrm{gas}$")
            if i == nrows - 1:
                ax.set_xlabel(r"$M_\star / M_\odot$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)

            if i < nrows - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(10**extent[2], 10**extent[3])
            ax.set_xlim(10**extent[0], 10**extent[1])

            # Label axis
            ax.text(0.95, 0.1, labels[i * ncols + j],
                    bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                              alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axes.append(ax)

    for (ind, t), l in zip(enumerate(types), labels):

        path = ini_path.replace("<type>", t)

        print(path)

        mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/ApertureMeasurements/Mass/030kpc",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 10
        part_mets = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                        snap,
                                        "PartType0/SmoothedMetallicity",
                                        noH=True, physicalUnits=True,
                                        numThreads=8)
        cops = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/CentreOfPotential",
                                   noH=True, physicalUnits=True,
                                   numThreads=8) * 1000
        grps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/GroupNumber",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)
        subgrps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                      snap,
                                      "Subhalo/SubGroupNumber",
                                      noH=True, physicalUnits=True,
                                      numThreads=8)
        ini_mass = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                       snap,
                                       "PartType0/Mass",
                                       noH=True, physicalUnits=True,
                                       numThreads=8) * 10 ** 10
        coords = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                     snap,
                                     "PartType0/Coordinates",
                                     noH=True, physicalUnits=True,
                                     numThreads=8) * 1000
        part_grps = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                        snap,
                                        "PartType0/GroupNumber",
                                        noH=True, physicalUnits=True,
                                        numThreads=8)
        part_subgrps = eagle_io.read_array("PARTDATA",
                                           path.replace("<type>", t),
                                           snap,
                                           "PartType0/SubGroupNumber",
                                           noH=True, physicalUnits=True,
                                           numThreads=8)

        # Apply some cuts
        mokinds = mass > 10**8
        mass = mass[mokinds]
        cops = cops[mokinds, :]
        grps = grps[mokinds]
        subgrps = subgrps[mokinds]

        # Set up array to store sfrs
        mets = []
        ms = []

        # Loop over galaxies
        for igal in range(mass.size):

            # Get galaxy data
            m = mass[igal]
            cop = cops[igal, :]
            g = grps[igal]
            sg = subgrps[igal]

            # Get this galaxies stars
            sokinds = np.logical_and(part_grps == g, part_subgrps == sg)
            this_coords = coords[sokinds, :] - cop
            this_ini_mass = ini_mass[sokinds]
            this_part_mets = part_mets[sokinds]

            # Compute stellar radii
            rs = np.sqrt(this_coords[:, 0] ** 2
                         + this_coords[:, 1] ** 2
                         + this_coords[:, 2] ** 2)

            # Get only particles within the aperture
            rokinds = rs < 30
            this_ini_mass = this_ini_mass[rokinds]
            this_part_mets = this_part_mets[rokinds]

            if len(this_ini_mass) == 0:
                continue

            # Store values for plotting
            ms.append(m)
            mets.append(np.average(this_part_mets, weights=this_ini_mass))

        # Convert to arrays and get mask
        mets = np.array(mets)
        ms = np.array(ms)
        okinds = mets > 0

        im = axes[ind].hexbin(ms[okinds], mets[okinds], mincnt=1, gridsize=50,
                              xscale="log", yscale="log", linewidth=0.2,
                              cmap="plasma", norm=norm, extent=extent)

    # Set up colorbar
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$N$")

    # Save figure
    mkdir("plots/massmet/")
    fig.savefig("plots/massmet/mass_gas_metallicity_%s.png" % snap,
                bbox_inches="tight")


def plot_stellarmet_mass_relation_evo_vary():

    # Define binning
    mass_bins = np.logspace(8.0, 13, 30)

    # Define the path
    ini_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["flares_00", "FLARES_00_REF",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh",
             "flares_00_no_agn", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "flares_00_H_reion_z03",
             "flares_00_H_reion_z075", "flares_00_H_reion_z14"]

    # Define labels for each
    labels = ["AGNdT9", "REF",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{r, 0}$", "$z_{r, 7.5}$",
              "$z_{r, 14}$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]

    # Define snapshots
    snaps = ["009_z006p000", "007_z008p000", "008_z007p000",
             "009_z006p000", "010_z005p000"]

    # Define plot dimensions
    nrows = 1
    ncols = len(snaps)

    # Define norm
    norm = LogNorm(vmin=1, vmax=10)

    # Define hexbin extent
    extent = [8, 11.5, -4, -1]

    # Set up the plots
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[1:, -1])

    for j in range(ncols):

        # Define redshift
        z = float(snaps[j].split("z")[-1].replace("p", "."))

        # Create axis
        ax = fig.add_subplot(gs[j])
        ax.loglog()

        # Include labels
        if j == 0:
            ax.set_ylabel(r"$Z_\star$")
        ax.set_xlabel(r"$M_\star / M_\odot$")

        # Remove unnecessary ticks
        if j > 0:
            ax.tick_params("y", left=False, right=False,
                           labelleft=False, labelright=False)

        # Set axis limits
        ax.set_ylim(10**extent[2], 10**extent[3])
        ax.set_xlim(10**extent[0], 10**extent[1])

        # Label axis
        ax.text(0.95, 0.9, "$z=$%d" % z,
                bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                          alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        axes.append(ax)

    for ax, snap in zip(axes, snaps):

        for (ind, t), l in zip(enumerate(types), labels):

            path = ini_path.replace("<type>", t)

            print(path)

            mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                       snap,
                                       "Subhalo/ApertureMeasurements/Mass/030kpc",
                                       noH=True, physicalUnits=True,
                                       numThreads=8)[:, 4] * 10 ** 10
            part_mets = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                            snap,
                                            "PartType4/SmoothedMetallicity",
                                            noH=True, physicalUnits=True,
                                            numThreads=8)
            cops = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                       snap,
                                       "Subhalo/CentreOfPotential",
                                       noH=True, physicalUnits=True,
                                       numThreads=8) * 1000
            grps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                       snap,
                                       "Subhalo/GroupNumber",
                                       noH=True, physicalUnits=True,
                                       numThreads=8)
            subgrps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                          snap,
                                          "Subhalo/SubGroupNumber",
                                          noH=True, physicalUnits=True,
                                          numThreads=8)
            ini_mass = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                           snap,
                                           "PartType4/InitialMass",
                                           noH=True, physicalUnits=True,
                                           numThreads=8) * 10 ** 10
            coords = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                         snap,
                                         "PartType4/Coordinates",
                                         noH=True, physicalUnits=True,
                                         numThreads=8) * 1000
            part_grps = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                            snap,
                                            "PartType4/GroupNumber",
                                            noH=True, physicalUnits=True,
                                            numThreads=8)
            part_subgrps = eagle_io.read_array("PARTDATA",
                                               path.replace("<type>", t),
                                               snap,
                                               "PartType4/SubGroupNumber",
                                               noH=True, physicalUnits=True,
                                               numThreads=8)

            # Apply some cuts
            mokinds = mass > 10**8
            mass = mass[mokinds]
            cops = cops[mokinds, :]
            grps = grps[mokinds]
            subgrps = subgrps[mokinds]

            # Set up array to store sfrs
            mets = []
            ms = []

            # Loop over galaxies
            for igal in range(mass.size):

                # Get galaxy data
                m = mass[igal]
                cop = cops[igal, :]
                g = grps[igal]
                sg = subgrps[igal]

                # Get this galaxies stars
                sokinds = np.logical_and(part_grps == g, part_subgrps == sg)
                this_coords = coords[sokinds, :] - cop
                this_ini_mass = ini_mass[sokinds]
                this_part_mets = part_mets[sokinds]

                # Compute stellar radii
                rs = np.sqrt(this_coords[:, 0] ** 2
                             + this_coords[:, 1] ** 2
                             + this_coords[:, 2] ** 2)

                # Get only particles within the aperture
                rokinds = rs < 30
                this_ini_mass = this_ini_mass[rokinds]
                this_part_mets = this_part_mets[rokinds]

                if len(this_ini_mass) == 0:
                    continue

                # Store values for plotting
                ms.append(m)
                mets.append(np.average(this_part_mets, weights=this_ini_mass))

            # Convert to arrays and get mask
            mets = np.array(mets)
            ms = np.array(ms)
            okinds = mets > 0

            plot_meidan_stat(ms[okinds], mets[okinds],
                             np.ones(mets[okinds].size),
                             ax, lab=l,
                             color=None, bins=mass_bins,
                             ls=linestyles[ind])

    # Draw legend
    axes[2].legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/massmet/")
    fig.savefig("plots/massmet/mass_stellar_metallicity_evo.png",
                bbox_inches="tight")


def plot_gasmet_mass_relation_evo_vary():

    # Define binning
    mass_bins = np.logspace(8.0, 13, 30)

    # Define the path
    ini_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["flares_00", "FLARES_00_REF",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh",
             "flares_00_no_agn", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "flares_00_H_reion_z03",
             "flares_00_H_reion_z075", "flares_00_H_reion_z14"]

    # Define labels for each
    labels = ["AGNdT9", "REF",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{r, 0}$", "$z_{r, 7.5}$",
              "$z_{r, 14}$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]

    # Define snapshots
    snaps = ["009_z006p000", "007_z008p000", "008_z007p000",
             "009_z006p000", "010_z005p000"]

    # Define plot dimensions
    nrows = 1
    ncols = len(snaps)

    # Define norm
    norm = LogNorm(vmin=1, vmax=10)

    # Define hexbin extent
    extent = [8, 11.5, -4, -1]

    # Set up the plots
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[1:, -1])

    for j in range(ncols):

        # Define redshift
        z = float(snaps[j].split("z")[-1].replace("p", "."))

        # Create axis
        ax = fig.add_subplot(gs[j])
        ax.loglog()

        # Include labels
        if j == 0:
            ax.set_ylabel(r"$Z_\mathrm{gas}$")
        ax.set_xlabel(r"$M_\star / M_\odot$")

        # Remove unnecessary ticks
        if j > 0:
            ax.tick_params("y", left=False, right=False,
                           labelleft=False, labelright=False)

        # Set axis limits
        ax.set_ylim(10**extent[2], 10**extent[3])
        ax.set_xlim(10**extent[0], 10**extent[1])

        # Label axis
        ax.text(0.95, 0.9, "$z=$%d" % z,
                bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                          alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        axes.append(ax)

    for ax, snap in zip(axes, snaps):

        for (ind, t), l in zip(enumerate(types), labels):

            path = ini_path.replace("<type>", t)

            print(path)

            mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                       snap,
                                       "Subhalo/ApertureMeasurements/Mass/030kpc",
                                       noH=True, physicalUnits=True,
                                       numThreads=8)[:, 4] * 10 ** 10
            part_mets = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                            snap,
                                            "PartType0/SmoothedMetallicity",
                                            noH=True, physicalUnits=True,
                                            numThreads=8)
            cops = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                       snap,
                                       "Subhalo/CentreOfPotential",
                                       noH=True, physicalUnits=True,
                                       numThreads=8) * 1000
            grps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                       snap,
                                       "Subhalo/GroupNumber",
                                       noH=True, physicalUnits=True,
                                       numThreads=8)
            subgrps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                          snap,
                                          "Subhalo/SubGroupNumber",
                                          noH=True, physicalUnits=True,
                                          numThreads=8)
            ini_mass = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                           snap,
                                           "PartType0/Mass",
                                           noH=True, physicalUnits=True,
                                           numThreads=8) * 10 ** 10
            coords = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                         snap,
                                         "PartType0/Coordinates",
                                         noH=True, physicalUnits=True,
                                         numThreads=8) * 1000
            part_grps = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                            snap,
                                            "PartType0/GroupNumber",
                                            noH=True, physicalUnits=True,
                                            numThreads=8)
            part_subgrps = eagle_io.read_array("PARTDATA",
                                               path.replace("<type>", t),
                                               snap,
                                               "PartType0/SubGroupNumber",
                                               noH=True, physicalUnits=True,
                                               numThreads=8)

            # Apply some cuts
            mokinds = mass > 10**8
            mass = mass[mokinds]
            cops = cops[mokinds, :]
            grps = grps[mokinds]
            subgrps = subgrps[mokinds]

            # Set up array to store sfrs
            mets = []
            ms = []

            # Loop over galaxies
            for igal in range(mass.size):

                # Get galaxy data
                m = mass[igal]
                cop = cops[igal, :]
                g = grps[igal]
                sg = subgrps[igal]

                # Get this galaxies stars
                sokinds = np.logical_and(part_grps == g, part_subgrps == sg)
                this_coords = coords[sokinds, :] - cop
                this_ini_mass = ini_mass[sokinds]
                this_part_mets = part_mets[sokinds]

                # Compute stellar radii
                rs = np.sqrt(this_coords[:, 0] ** 2
                             + this_coords[:, 1] ** 2
                             + this_coords[:, 2] ** 2)

                # Get only particles within the aperture
                rokinds = rs < 30
                this_ini_mass = this_ini_mass[rokinds]
                this_part_mets = this_part_mets[rokinds]

                if len(this_ini_mass) == 0:
                    continue

                # Store values for plotting
                ms.append(m)
                mets.append(np.average(this_part_mets, weights=this_ini_mass))

            # Convert to arrays and get mask
            mets = np.array(mets)
            ms = np.array(ms)
            okinds = mets > 0

            plot_meidan_stat(ms[okinds], mets[okinds],
                             np.ones(mets[okinds].size),
                             ax, lab=l,
                             color=None, bins=mass_bins,
                             ls=linestyles[ind])

    # Draw legend
    axes[2].legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/massmet/")
    fig.savefig("plots/massmet/mass_gas_metallicity_evo.png",
                bbox_inches="tight")


if __name__ == "__main__":
    plot_stellarmet_mass_relation_vary("010_z005p000")
    plot_gasmet_mass_relation_vary("010_z005p000")
    plot_stellarmet_mass_relation_evo_vary()
    plot_gasmet_mass_relation_evo_vary()
