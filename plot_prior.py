import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

deg = np.pi / 180.0

def p(u, mu, tau):
    v = np.log(u / (1 - u))
    return 1/(u * (1-u)) * np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (v - mu)**2)

np.random.seed(42)

N = 50
taus = np.abs(np.random.normal(loc=0., scale=4.0, size=N)) + 0.5
mus = np.random.normal(loc=0., scale=2.0, size=N)

us = np.linspace(1e-04, 1 - 1e-04, num=500)

ps = [p(us, mu, tau) for (mu, tau) in zip(mus, taus)]
# normalize to peak value
pn = [p/np.max(p) for p in ps]

fig, ax = plt.subplots(nrows=1, figsize=(3.5, 1.8))

for i in range(N):
    ax.plot(us * 180.0, pn[i], lw=0.8, color="C0", alpha=0.7)

ax.set_xlabel(r"$\theta\; [{}^\circ]$")
ax.set_ylabel(r"$p(\theta)$")
ax.yaxis.set_ticklabels([])
ax.set_xlim(0, 180)
ax.set_ylim(0, 1.02)
fig.subplots_adjust(bottom=0.25, top=0.98, left=0.09, right=0.91)
fig.savefig("prior.pdf")
fig.savefig("prior.png", dpi=600)



# let's also plot some blank figures and the isotropic and low-prior for kicks
fig, ax = plt.subplots(nrows=1, figsize=(3.5, 1.8))
ax.set_xlabel(r"$\theta\; [{}^\circ]$")
ax.set_ylabel(r"$p(\theta)$")
ax.yaxis.set_ticklabels([])
ax.set_xlim(0, 180)
ax.set_ylim(0, 1.02)
fig.subplots_adjust(bottom=0.25, top=0.98, left=0.09, right=0.91)
fig.savefig("prior_blank.pdf")
fig.savefig("prior_blank.png", dpi=600)


# let's also plot some blank figures and the isotropic and low-prior for kicks
fig, ax = plt.subplots(nrows=1, figsize=(3.5, 1.8))

dist = np.sin(us * np.pi)/2 * deg
dist = dist / np.max(dist)

ax.plot(us * 180.0, dist, lw=2, color="C0") #, alpha=0.7)

ax.set_xlabel(r"$\theta\; [{}^\circ]$")
ax.set_ylabel(r"$p(\theta)$")
ax.yaxis.set_ticklabels([])
ax.set_xlim(0, 180)
ax.set_ylim(0, 1.02)
fig.subplots_adjust(bottom=0.25, top=0.98, left=0.09, right=0.91)
fig.savefig("prior_sin.pdf")
fig.savefig("prior_sin.png", dpi=600)



fig, ax = plt.subplots(nrows=1, figsize=(3.5, 1.8))

dist = p(us, -3.0, 3.5)
dist = dist / np.max(dist)
ax.plot(us * 180.0, dist, lw=2, color="C0") #, alpha=0.7)

ax.set_xlabel(r"$\theta\; [{}^\circ]$")
ax.set_ylabel(r"$p(\theta)$")
ax.yaxis.set_ticklabels([])
ax.set_xlim(0, 180)
ax.set_ylim(0, 1.02)
fig.subplots_adjust(bottom=0.25, top=0.98, left=0.09, right=0.91)
fig.savefig("prior_low.pdf")
fig.savefig("prior_low.png", dpi=600)
