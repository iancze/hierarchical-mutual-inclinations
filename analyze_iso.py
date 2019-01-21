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


deg = np.pi/180.0

# load the dataset
data = ascii.read("data/iso_sample.ecsv", format="ecsv")
N_systems = len(data)

# instantiate a PyMC3 model class
with pm.Model() as model:

    log_alpha = pm.Uniform("log_alpha", lower=0.01, upper=2.5)
    log_beta = pm.Uniform("log_beta", lower=0.01, upper=2.5)

    v = pm.Beta("v", alpha=10**log_alpha, beta=10**log_beta, shape=N_systems)

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

# plot = pm.traceplot(trace)
# plt.savefig("iso/trace.png")

# For fun, let's see how well the model inferred the distribution of \phi angles as well, even though this
# was not something we observed, we still have posterior probability distributions of them

fig = plt.figure(figsize=(6.5,4))
gs = gridspec.GridSpec(nrows=3, ncols=4, figure=fig, left=0.07, right=0.93, top=0.98, bottom=0.11, hspace=0.3) #left, right, etc..

# the mutual inc dis
ax_mut = plt.subplot(gs[0, 0:2])
ax_mut.set_ylabel(r"$p(\theta| \,\boldsymbol{D})\quad[{}^\circ$]")
ax_mut.yaxis.set_ticklabels([])
ax_mut.annotate(r"$\theta$", (0.9,0.8), xycoords="axes fraction")

nplot = 20
ind = np.random.choice(range(len(trace)), nplot)
log_alphas = trace["log_alpha"][ind]
log_betas = trace["log_beta"][ind]

vs = np.linspace(0, 1, num=200)
for i in range(nplot):
    ys = beta.pdf(vs, 10**log_alphas[i], 10**log_betas[i])/np.pi * deg
    ax_mut.plot(vs * 180., ys, lw=0.8, alpha=0.8, color="C0")


# norm = quad(input_prob, 0, 180.)
# print(norm)
# plot the real distribution
ax_mut.plot(vs * 180, np.sin(vs * np.pi)/2 * deg, lw=1.5, color="k")
#
# res = quad(lambda x: beta.pdf(x/np.pi, 1.2, 10.0)/np.pi, 0, np.pi)
# print(res)

# ys = beta.pdf(vs, 3.0, 50.0)/np.pi * deg
# ax_mut.plot(vs * 180., ys, lw=0.8, alpha=0.8, color="C1")
#
# ys = beta.pdf(vs, 2.0, 100.0)/np.pi * deg
# ax_mut.plot(vs * 180., ys, lw=0.8, alpha=0.8, color="C1")

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
ax[6].set_ylabel(r"$p(\theta_j| \,\boldsymbol{D})\quad[{}^\circ$]")

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

fig.savefig("iso/imut_iso.pdf")
