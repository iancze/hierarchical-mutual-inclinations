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


def calc_theta(i_disk, Omega_disk, i_star, Omega_star):
    '''
    Calculate the mutual inclination between two planes. Assumes all inputs in degrees.
    '''

    cos_theta = np.cos(i_disk * deg) * np.cos(i_star * deg) +         np.sin(i_disk * deg) * np.sin(i_star * deg) * np.cos((Omega_disk - Omega_star) * deg)
    theta = np.arccos(cos_theta)/deg

    return theta

# Rejection sampling for the distributions
def input_prob(theta):
    '''
    theta is in degrees.
    '''
    return np.sin(theta * deg) * np.exp(-0.5 * (theta - 5.0)**2 / (2**2))

def propose_prob(theta):
    '''
    theta is in degrees.
    '''
    return np.exp(-0.5 * (theta - 5.0)**2/(2**2))


# proposed values
theta_props = np.random.normal(loc=5.0, scale=2.0, size=100000)

# since we know that the mutual inclination must be between 0 and 180, let's get rid of any values on the tail
# of the Gaussian that exceed these ranges
theta_props = theta_props[(theta_props > 0.0) & (theta_props < 180.0)]

# Evaluate the probability of Q for all samples
Qs = propose_prob(theta_props)

# Generate a random number [0, Q(Delta I)] for all samples
us = np.random.uniform(low=0, high=Qs)

# Evaluate p(Delta I) for all samples
Ps = input_prob(theta_props)

# Keep the samples for which u <= P
theta_samples = theta_props[us <= Ps]


thetas = np.random.choice(theta_samples, size=N_systems) # degrees
phis = np.random.uniform(low=0, high=360., size=N_systems) # degrees


def P_x(xi):
    '''
    Calculate the transformation matrix about the X-axis. Assumes the angle is in degrees.
    '''
    x = xi * deg
    return np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])

def P_z(xi):
    '''
    Calculate the transformation matrix about the Z-axis. Assumes angle is in degrees.
    '''
    x = xi * deg
    return np.array([[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]])

def get_xyz(theta, phi):
    '''
    Convert angles in the disk frame into x, y, z in the disk frame. All angles in degrees.
    '''
    return np.array([np.sin(theta * deg) * np.cos(phi * deg), np.sin(theta * deg) * np.sin(phi * deg),                      np.cos(theta * deg)])


def get_binary_parameters(i_disk, Omega_disk, theta, phi):
    '''
    Using the matrix math above, calculate i_star and Omega_star. Assumes all angles are in degrees.
    '''

    xyz = get_xyz(theta, phi)

    res = np.dot(P_z(-Omega_disk).T, np.dot(P_x(i_disk).T, xyz))

    X, Y, Z = res

    i_star = np.arccos(Z)/deg
    Omega_star = np.arctan2(Y, X)/deg - 90.0
    # the -90 is because the projection of the angular momentum vector on the X-Y plane
    # is not the same as the ascending node. There is a 90 degree offset between the two.

    return i_star, Omega_star

i_stars = np.empty(N_systems, np.float64) # degrees
Omega_stars = np.empty(N_systems, np.float64) # degrees

for i in range(N_systems):
    i_stars[i], Omega_stars[i] = get_binary_parameters(i_disks[i], Omega_disks[i], thetas[i], phis[i]);

# calculate the uncertainty assuming 1 degree and save this too
# we'll assume that the uncertainties on the fake data are a degree for both the disk and the binary
# and the error is on cos_i()
sd_disks = np.sin(i_disks * deg) * 1.0 * deg
sd_stars = np.sin(i_stars * deg) * 1.0 * deg

# compute the mutual inclination of these too, and add it as a column
# for fun, also store the phi column, since our posterior will have it
low_sample = Table([np.cos(i_disks * deg), sd_disks, Omega_disks, np.cos(i_stars * deg), sd_stars, Omega_stars, thetas, phis], names=["cos_i_disk", "cos_i_disk_err", "Omega_disk", "cos_i_star", "cos_i_star_err", "Omega_star", "theta", "phi"])

ascii.write(low_sample, "data/low_sample.ecsv", format="ecsv")
