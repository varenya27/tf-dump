import emcee
import numpy as np
from scipy.odr import Model, RealData, ODR
import matplotlib as mpl
from matplotlib import pyplot as plt# import pyplot from matplotlib
from time import time               # use for timing functions
import corner

def straight_line(theta,x):

    y=theta[0]*x + theta[1]
    return y

# plot the resulting posteriors
mpl.rcParams.update({'font.size': 16})

min_ = -10.
max_ = 10.
min_scat = -5.
max_scat = 2.
def logprior(theta):
    lp = 0.
    m, c, scat_int = theta


    # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
    # (we don't care about it being properly normalised, but you can if you want) 
    if scat_int<0: return -np.inf
    lp = 0. if min_ < c < max_ and min_ < m < max_ and min_scat < np.log(scat_int) < max_scat else -np.inf

    # Gaussian prior on m
    mmu = 0.     # mean of the Gaussian prior
    msigma = 10. # standard deviation of the Gaussian prior
    lp -= 0.5 * ((m - mmu) / msigma)**2

    return lp

def loglikelihood(theta, y, x, err_y, err_x):
    # unpack the model parameters from the tuple
    m, c, sigma_int = theta
    sigma2 = (m**2*err_x**2)/(m**2+1)+(m**2*err_y**2)/(m**2+1)+sigma_int**2
    # evaluate the model (assumes that the straight_line model is defined as above)
    md = straight_line(theta,x)

    delta = ( (y-md)**2) / (m**2+1)
    # return the log likelihood
    return -0.5 * np.sum(np.log(2*np.pi*sigma2)+(delta/(sigma2)))

def logposterior(theta, y, x, err_y, err_x):
    lp = logprior(theta) # get the prior
    # if the prior is not finite return a probability of zero (log probability of -inf)
    if not np.isfinite(lp):
        return -np.inf

    # return the likeihood times the prior (log likelihood plus the log prior)
    return lp + loglikelihood(theta, y, x, err_y, err_x)

Nens = 100   # number of ensemble points
ndims = 3
Nburnin = 500   # number of burn-in samples
Nsamples = 500  # number of final posterior samples


y, err_y, x, err_x= np.loadtxt("values1.txt", unpack=True)
# print(y,x)
argslist = (y, x, err_y, err_x)

model = Model(straight_line)
data = RealData(x, y, err_x, err_y)
odr = ODR(data, model, [1,1])
odr.set_job(fit_type=0)
output = odr.run()
a_ODR = output.beta[0] #slope
b_ODR = output.beta[1] #normalization
err_a_ODR = output.sd_beta[0] #error on slope
err_b_ODR = output.sd_beta[1] #error on norm
res_ODR = (y - output.beta[0]*x - output.beta[1]) #residuals
rms_ODR = np.sqrt(np.mean(res_ODR**2., dtype=np.float64)) #observed scatter
s_ODR = np.sqrt(rms_ODR**2. - np.mean(err_y)**2.) #estimate of intrinsic scatter

p0 = []
for i in range(Nens):
    pi = [
        np.random.uniform(a_ODR-20*err_a_ODR,a_ODR+20*err_a_ODR), 
        np.random.uniform(b_ODR-20*err_b_ODR,b_ODR+20*err_b_ODR),
        np.random.uniform(rms_ODR/10., rms_ODR*10.)]
    # pi = [np.random.uniform(2, 4), np.random.uniform(1, 3), np.random.uniform(rms_ODR/10., rms_ODR)]
    # pi = [np.random.normal(a_ODR, err_a_ODR), np.random.normal(b_ODR, err_b_ODR), np.random.uniform(rms_ODR/10., rms_ODR)]
    p0.append(pi)

# set up the sampler
sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)
# pass the initial samples and total number of samples required
t0 = time() # start time
sampler.run_mcmc(p0, Nsamples + Nburnin);
t1 = time()

timeemcee = (t1-t0)
print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

# extract the samples (removing the burn-in)
samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)


#plots
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
m_final = np.percentile(flat_samples[:, 0], [16, 50, 84])[1]
c_final = np.percentile(flat_samples[:, 1], [16, 50, 84])[1]
scat_final = np.percentile(flat_samples[:, 2], [16, 50, 84])[1]
Med_value = [m_final,c_final,scat_final]

figure = corner.corner(
    flat_samples,
    # title_fmt=".2E",
    levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), 
    labels=[r"Slope", r"Intercept", r"Intrinsic Scatter", r"Intrinsic Scatter"], 
    quantiles=[0.16,0.84], 
    range=[(m_final-0.8,m_final+0.8), (c_final-1.6,c_final+1.6), (scat_final-0.03,scat_final+0.03)],
    show_titles=True, 
    label_kwargs={"fontsize": 8},
    title_kwargs={"fontsize": 10}
    
);

axes = np.array(figure.axes).reshape((ndims, ndims))
for i in range(ndims):
    ax = axes[i, i]
    ax.axvline(Med_value[i], color="r")
for yi in range(ndims):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(Med_value[xi], color="r")
        ax.axhline(Med_value[yi], color="r")
        ax.plot(Med_value[xi], Med_value[yi], "sr")
figure.savefig('corner.pdf',format='pdf', dpi=300)
# figure = corner.corner(samples, levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), labels=[r"Slope", r"Intercept", r"Intrinsic Scatter", r"Intrinsic Scatter"], quantiles=[0.16,0.84], show_titles=True, label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 10}, range=[(a_ML-0.4,a_ML+0.4), (b_ML-0.6,b_ML+0.6), (s_ML-0.1,s_ML+0.1)])


for i in range(ndims):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(mcmc[1], q[0], q[1])
    # display(Math(txt))