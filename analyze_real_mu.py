import numpy as np
import scipy.stats
from scipy.integrate import quad
import matplotlib
# matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pymc3 as pm
import theano.tensor as tt

from astropy.io import ascii


deg = np.pi/180.0

# load the dataset
data = ascii.read("data/real_sample.ecsv", format="ecsv")
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
    trace = pm.backends.ndarray.load_trace("real_logit")
#
# plot = pm.traceplot(trace)
# plt.savefig("real_logit/trace.png")

# For fun, let's see how well the model inferred the distribution of \phi angles as well, even though this
# was not something we observed, we still have posterior probability distributions of them

fig = plt.figure(figsize=(3.25,6))
gs = gridspec.GridSpec(nrows=4, ncols=2, figure=fig, left=0.11, right=0.89, top=0.98, bottom=0.11, hspace=0.3, wspace=0.15) #left, right, etc..

# the mutual inc dis
ax_mut = plt.subplot(gs[0, :])
ax_mut.set_ylabel(r"$p(\theta|\,\boldsymbol{D})\quad[{}^\circ$]")
ax_mut.yaxis.set_ticklabels([])
ax_mut.annotate(r"$\theta$", (0.9,0.8), xycoords="axes fraction")
ax_mut.set_xlim(0, 15)

np.random.seed(42)
nplot = 20
ind = np.random.choice(range(len(trace)), nplot)
mus = trace["mu"][ind]
taus = trace["tau_off"][ind]

us = np.linspace(0.001, 30/180, num=500)
vs = np.log(us/(1 - us))
for i in range(nplot):
    ys = 1/(us * (1 - us)) * np.sqrt(taus[i]/(2 * np.pi)) * np.exp(-taus[i]/2 * (vs - mus[i])**2)/np.pi * deg
    # ax_mut.plot(us * 180., ys/np.max(ys), lw=0.8, alpha=0.8, color="C0")
    ax_mut.plot(us * 180., ys/np.max(ys), lw=0.8, alpha=0.8, color="C0")

ax_mut_bin = plt.subplot(gs[1, :])

# the individual mutual inclinations
ax = [
plt.subplot(gs[2,0]),
plt.subplot(gs[2,1]),
plt.subplot(gs[3,0]),
plt.subplot(gs[3,1]),]

ax[2].set_xlabel(r"$\theta_j\quad[{}^\circ$]")
ax[2].set_ylabel(r"$p(\theta_j|\,\boldsymbol{D})\quad[{}^\circ$]")

theta_samples = trace["theta"]

xlim = (0,15)
nbins = 40
bins = np.linspace(*xlim, num=nbins)

labels = [r"V4046\;Sgr", r"AK\;Sco", r"DQ\;Tau", r"UZ\;Tau\;E"]

for i,a in enumerate(ax):
    heights, b, patches = a.hist(theta_samples[:,i], bins=bins, density=True)
    a.yaxis.set_ticklabels([])
    a.set_xlim(*xlim)
    a.annotate(r"$\theta_\mathrm{" + labels[i] + r"}$", (0.45,0.8), xycoords="axes fraction")

    dx = b[1] - b[0]
    tot_prob = np.cumsum(heights * dx)
    ind = np.searchsorted(tot_prob, 0.683)
    print("{:} 68 percentile".format(labels[i]), b[1:][ind])

    # also calculate the numbers for how much of the probability is below 68%

ax_mut.set_xlim(0, 30)
ax_mut.set_ylim(bottom=0, top=1.05)

# do it again for the full dist

npoints = 200
points = np.linspace(0.001, 30, num=npoints)

# make a marginalizied plot over the bins we chose
# sample the draws of the logit-normal distribution
nsample = 10000
ind = np.random.choice(range(len(trace)), nsample)
mus = trace["mu"][ind]
taus = trace["tau_off"][ind]

upoints = points/180.0
vpoints = np.log(upoints/(1 - upoints))
ypoints = np.empty((nsample, npoints))

for i in range(nsample):
    ypoints[i] = 1/(upoints * (1 - upoints)) * np.sqrt(taus[i]/(2 * np.pi)) * np.exp(-taus[i]/2 * (vpoints - mus[i])**2)/np.pi * deg

final = np.average(ypoints, axis=0)
# get the 68% draws
bounds = np.percentile(ypoints, [50 - 34.1, 50+34.1], axis=0) # compute the bounding regions at each point
# stdev = np.std(ypoints, axis=0)
# print(len(stdev))
# print()

# do this for the actual distribution
dx = points[1] - points[0]
tot_prob = np.cumsum(final * dx)
print(tot_prob[-1])
ind = np.searchsorted(tot_prob/tot_prob[-1], 0.683)
print("68 percentile all", points[1:][ind])

ind = np.searchsorted(tot_prob/tot_prob[-1], 0.954)
print("95.4 percentile all", points[1:][ind])

ind = np.searchsorted(tot_prob/tot_prob[-1], 0.973)
print("97.3 percentile all", points[1:][ind])


ax_mut_bin.fill_between(points, bounds[0], bounds[1], alpha=0.2, color="C0", edgecolor=None, linewidth=0.0)
ax_mut_bin.plot(points, final, color="C0")
# ax_mut_bin.plot(points, bounds[0], color="C1")
# ax_mut_bin.plot(points, bounds[1], color="C1")

ax_mut_bin.set_ylabel(r"$\langle p(\theta|\,\boldsymbol{D}) \rangle \quad[{}^\circ$]")
ax_mut_bin.yaxis.set_ticklabels([])
ax_mut_bin.set_xlim(0, 30)
ax_mut_bin.set_ylim(bottom=0.0)

fig.savefig("real_logit/imut_real.pdf")
fig.savefig("real_logit/imut_real.png")
