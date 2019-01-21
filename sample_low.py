import matplotlib.pyplot as plt

import numpy as np
import theano.tensor as tt
import pymc3 as pm
import corner

from astropy.io import ascii

deg = np.pi/180.0

# load the dataset
data = ascii.read("data/low_sample.ecsv", format="ecsv")
N_systems = len(data)

# instantiate a PyMC3 model class
with pm.Model() as model:

    mu = pm.Normal("mu", mu=0.0, sd=2.0)
    tau = pm.HalfNormal("tau", sd=4.0)
    tau_off = pm.Deterministic("tau_off", tau + 0.5)

    v = pm.LogitNormal("v", mu=mu, tau=tau_off, shape=N_systems)

    theta = pm.Deterministic("theta", v * 180.)

    cos_theta = tt.cos(v * np.pi)
    sin_theta = tt.sin(v * np.pi)

    # Enforce the geometrical prior on i_disk, as before
    # Testval tells the chain to start in the center of the posterior.
    cos_i_disk = pm.Uniform("cosIdisk", lower=-1.0, upper=1.0, shape=N_systems, testval=data["cos_i_disk"].data)

    sin_i_disk = tt.sqrt(1.0 - cos_i_disk**2)

    # This is the azimuthal orientation of the binary vector on the circle some theta away
    phi = pm.Uniform("phi", lower=-np.pi/2.0, upper=np.pi/2.0, shape=N_systems)

    cos_i_star = pm.Deterministic("cos_i_star", -sin_i_disk * sin_theta * tt.sin(phi) + cos_i_disk * cos_theta)

    # Finally, we define the likelihood by conditioning on the observations using a Normal
    obs_disk = pm.Normal("obs_disk", mu=cos_i_disk, sd=data["cos_i_disk_err"].data, observed=data["cos_i_disk"].data)
    obs_star = pm.Normal("obs_star", mu=cos_i_star, sd=data["cos_i_star_err"].data, observed=data["cos_i_star"].data)

# sample the model!
with model:
    trace = pm.sample(draws=10000, tune=40000, chains=2, nuts_kwargs={"target_accept":0.95})

pm.backends.ndarray.save_trace(trace, directory="low", overwrite=True)
