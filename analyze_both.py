import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
import matplotlib
# matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pymc3 as pm
import theano.tensor as tt

from astropy.io import ascii

np.random.seed(43)

deg = np.pi/180.0

# load the dataset
data = ascii.read("data/iso_sample.ecsv", format="ecsv")
N_systems = len(data)

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

with model:
    trace = pm.backends.ndarray.load_trace("iso")# , model=None)


fig = plt.figure(figsize=(6.5,8))
ax_frac = 0.42
top = 0.99
bottom = 0.06
gs = gridspec.GridSpec(nrows=3, ncols=4, figure=fig, left=0.065, right=0.935, top=top, bottom=top - ax_frac, hspace=0.3) #left, right, etc..
gs_low = gridspec.GridSpec(nrows=3, ncols=4, figure=fig, left=0.065, right=0.935, top=bottom+ax_frac, bottom=bottom, hspace=0.3) #left,

ax_line = fig.add_axes([0.0, 0.485, 1, 0.05]) # create an axes over the full figure.
ax_line.axis("off")
ax_line.plot([0.03, 0.97], [0.0, 0.0], color="k")
ax_line.set_xlim(0, 1)
ax_line.set_ylim(-1, 1)

# the mutual inc dis
ax_mut = plt.subplot(gs[0, 0:2])
ax_mut.set_ylabel(r"$p(\theta| \,\boldsymbol{D}_\mathrm{iso})\quad[{}^\circ$]")
ax_mut.yaxis.set_ticklabels([])
ax_mut.annotate(r"$\theta$", (0.9,0.8), xycoords="axes fraction")

nplot = 20
ind = np.random.choice(range(len(trace)), nplot)
mus = trace["mu"][ind]
taus = trace["tau_off"][ind]

us = np.linspace(0.001, 0.999, num=500)
vs = np.log(us/(1 - us))
for i in range(nplot):
    ys = 1/(us * (1 - us)) * np.sqrt(taus[i]/(2 * np.pi)) * np.exp(-taus[i]/2 * (vs - mus[i])**2)/np.pi * deg
    ax_mut.plot(us * 180., ys/np.max(ys), lw=0.8, alpha=0.8, color="C0")

# norm = quad(input_prob, 0, 180.)
# print(norm)
# plot the real distribution
dist = np.sin(vs * np.pi)/2 * deg
ax_mut.plot(vs * 180, dist/np.max(dist) , lw=1.7, color="w")
ax_mut.plot(vs * 180, dist/np.max(dist) , lw=1.5, color="k")

# the individual mutual inclinations
ax = [plt.subplot(gs[0, 2]), #1
plt.subplot(gs[0, 3]),
plt.subplot(gs[1,0]),
plt.subplot(gs[1,1]),
plt.subplot(gs[1,2]),
plt.subplot(gs[1,3]),
plt.subplot(gs[2,0]),
plt.subplot(gs[2,1]),
plt.subplot(gs[2,2]),
plt.subplot(gs[2,3])]

ax[6].set_xlabel(r"$\theta_j\quad[{}^\circ$]")
ax[6].set_ylabel(r"$p(\theta_j| \,\boldsymbol{D}_\mathrm{iso})\quad[{}^\circ$]")

theta_samples = trace["theta"]
thetas = data["theta"]
#
xlim = (0,180)

for i,a in enumerate(ax):
    a.hist(theta_samples[:,i], bins=50, density=True)
    a.axvline(thetas[i], color="k")
    a.yaxis.set_ticklabels([])
    a.set_xlim(*xlim)
    a.annotate(r"$\theta_{:}$".format(i), (0.8,0.8), xycoords="axes fraction")

ax_mut.set_xlim(*xlim)
ax_mut.set_ylim(bottom=0.0, top=1.05)


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

with model:
    trace = pm.backends.ndarray.load_trace("low")# , model=None)

# the mutual inc dis
ax_mut = plt.subplot(gs_low[0, 0:2])
ax_mut.set_ylabel(r"$p(\theta|\,\boldsymbol{D}_\mathrm{low})\quad[{}^\circ$]")
ax_mut.yaxis.set_ticklabels([])
ax_mut.annotate(r"$\theta$", (0.9,0.8), xycoords="axes fraction")

nplot = 20
ind = np.random.choice(range(len(trace)), nplot)
mus = trace["mu"][ind]
taus = trace["tau_off"][ind]

us = np.linspace(0.001, 30/180., num=500)
vs = np.log(us/(1 - us))
for i in range(nplot):
    ys = 1/(us * (1 - us)) * np.sqrt(taus[i]/(2 * np.pi)) * np.exp(-taus[i]/2 * (vs - mus[i])**2)/np.pi * deg
    ax_mut.plot(us * 180., ys/np.max(ys), lw=0.8, alpha=0.8, color="C0")

# Rejection sampling for the distributions
def input_prob(theta):
    '''
    theta is in degrees.
    '''
    return np.abs(np.sin(theta * deg)) * np.exp(-0.5 * (theta - 5.0)**2 / (2**2))

norm = quad(input_prob, 0, 180.)
print(norm)
# plot the real distribution
dist = input_prob(us * 180)/norm[0]
ax_mut.plot(us * 180, dist/np.max(dist), lw=1.7, color="w")
ax_mut.plot(us * 180, dist/np.max(dist), lw=1.5, color="k")

# the individual mutual inclinations
ax = [plt.subplot(gs_low[0, 2]), #1
plt.subplot(gs_low[0, 3]),
plt.subplot(gs_low[1,0]),
plt.subplot(gs_low[1,1]),
plt.subplot(gs_low[1,2]),
plt.subplot(gs_low[1,3]),
plt.subplot(gs_low[2,0]),
plt.subplot(gs_low[2,1]),
plt.subplot(gs_low[2,2]),
plt.subplot(gs_low[2,3])]

ax[6].set_xlabel(r"$\theta_j\quad[{}^\circ$]")
ax[6].set_ylabel(r"$p(\theta_j|\,\boldsymbol{D}_\mathrm{low})\quad[{}^\circ$]")

theta_samples = trace["theta"]
thetas = data["theta"]
#
xlim = (0,30)

for i,a in enumerate(ax):
    a.hist(theta_samples[:,i], bins=50, density=True)
    a.axvline(thetas[i], color="k")
    a.yaxis.set_ticklabels([])
    a.set_xlim(*xlim)
    a.annotate(r"$\theta_{:}$".format(i), (0.8,0.8), xycoords="axes fraction")


ax_mut.set_xlim(*xlim)
ax_mut.set_ylim(0,1.05)

# necessary position to plot on the side
# plt.figtext(0.96, 0.27, r"low $\theta$ sample", rotation="vertical", va="center", ha="center")
# plt.figtext(0.96, 0.83, r"isotropic sample", rotation="vertical", ha="center")

plt.figtext(0.97, 0.492, r"low $\theta$ sample", va="center", ha="right")
plt.figtext(0.97, 0.52, r"isotropic sample", ha="right")


fig.savefig("imut_fake.pdf")
