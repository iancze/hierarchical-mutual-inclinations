import numpy as np
from astropy.table import Table
from astropy.io import ascii

deg = np.pi/180.0

# np.random.seed(42)
np.random.seed(43)
N_systems = 10

# let's get a sample of uniformly oriented disks in 3D space
# Generate randomly oriented momentum vector on the sky.
U = np.random.uniform(size=N_systems)
V = np.random.uniform(size=N_systems)

# spherical coordinates for angular momentum vector, uniformly distributed.
i_disks = np.arccos(2 * V - 1.0) / deg # polar angle
Omega_disks = 2 * np.pi * U / deg # azimuth


# draw new random numbers for the binary
U = np.random.uniform(size=N_systems)
V = np.random.uniform(size=N_systems)

# spherical coordinates for angular momentum vector, uniformly distributed.
i_stars = np.arccos(2 * V - 1.0) / deg # polar angle
Omega_stars = 2 * np.pi * U / deg # azimuth


def calc_theta(i_disk, Omega_disk, i_star, Omega_star):
    '''
    Calculate the mutual inclination between two planes. Assumes all inputs in degrees.
    '''

    cos_theta = np.cos(i_disk * deg) * np.cos(i_star * deg) +         np.sin(i_disk * deg) * np.sin(i_star * deg) * np.cos((Omega_disk - Omega_star) * deg)
    theta = np.arccos(cos_theta)/deg

    return theta


thetas = calc_theta(i_disks, Omega_disks, i_stars, Omega_stars)

# calculate the uncertainty assuming 1 degree and save this too
# we'll assume that the uncertainties on the fake data are a degree for both the disk and the binary
# and the error is on cos_i()
sd_disks = np.sin(i_disks * deg) * 1.0 * deg
sd_stars = np.sin(i_stars * deg) * 1.0 * deg

# compute the mutual inclination of these too, and add it as a column
iso_sample = Table([np.cos(i_disks * deg), sd_disks, Omega_disks, np.cos(i_stars * deg), sd_stars, Omega_stars, thetas], names=["cos_i_disk", "cos_i_disk_err", "Omega_disk", "cos_i_star", "cos_i_star_err", "Omega_star", "theta"])

ascii.write(iso_sample, "data/iso_sample.ecsv", format="ecsv")
