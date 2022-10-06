import emcee
from matplotlib import pyplot as plt 
import numpy as np
from scipy.optimize import minimize
def log_likelihood(theta, x, y, yerr):
    x=np.array(x)
    y=np.array(y)
    yerr=np.array(yerr)
    m, b, log_f = theta
    print(type(m),b,log_f, theta)
    model = m * x + b
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    m, b, log_f = theta
    if 3.0 < m < 5.0 and 1.0 < b < 5.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

m_true = 3.85
b_true = 1.99
f_true = 0.026

m,m_err,vf,vf_err=[],[],[],[]
with open('codes/data.txt','r') as f:
    for line in f:
        line=line.split()
        m.append(float(line[1]))
        m_err.append(float(line[2]))
        vf.append(float(line[3]))
        vf_err.append(float(line[4]))

nll = lambda *args: -log_likelihood(*args)
print(nll)
initial = np.array([m_true, b_true, np.log(f_true)]) + 0.1 * np.random.randn(3)
print(initial)
soln = minimize(nll, initial, args=(vf, m, m_err))
print(soln)
m_ml, b_ml, log_f_ml = soln.x


pos = soln.x 
nwalkers, ndim = 50,3

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(vf, m, m_err)
)
sampler.run_mcmc(pos, 5000, progress=True);

fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b", "log(f)"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

fig = corner.corner(
    flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)]
);

inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", label="truth")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    txt = print(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))



plt.show()

