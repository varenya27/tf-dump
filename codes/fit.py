from scipy.odr import Model, RealData, ODR
import numpy as np
import emcee

def func(beta,x):
    return beta[0]*x+beta[1]

def log_prior(theta):
    m,b,log_s=theta
    if 0<m<5 and 0<b<5 and -5<log_s<0:
        return 0. 
    return -np.inf

def log_likelihood(theta,y,x,err_y,err_x):
    m,b,log_s=theta
    
    sum1,sum2=0,0
    for (i,j) in zip(err_x,err_y):
        delta = np.sqrt( (m**2)*(i**2) + j**2 +(np.exp(log_s))**2)
        sum1+=np.log(np.sqrt(2*np.pi)*delta)
    for(ix,iy,jx,jy) in zip(x,y,err_x, err_y):
        delta = np.sqrt( (m**2)*(jx**2) + jy**2 +(np.exp(log_s))**2)
        de = iy-(m*ix+b)
        sum2+= de**2/(2*delta**2)
    return -sum1-sum2

def log_probability(theta,y,x,err_y,err_x):
    m,b,log_s=theta
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp+log_likelihood(theta, y, x, err_y, err_x)

y, err_y, x, err_x= np.loadtxt("BayesLineFit/values.txt", unpack=True)
model = Model(func)
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
print(a_ODR, b_ODR,s_ODR)

m, b, log_s= a_ODR, b_ODR,np.log(s_ODR)
ndim = 3
nwalkers=50
max_iters=10000
pos=(m,b,log_s)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[y, x, err_y, err_x],)
p0 = []
for i in range(nwalkers):
    pi = [np.random.normal(a_ODR, err_a_ODR), np.random.normal(b_ODR, err_b_ODR), np.random.uniform(rms_ODR/10., rms_ODR)]
    p0.append(pi)
index = 0
autocorr = np.empty(max_iters)
old_tau = np.inf

for sample in sampler.sample(p0, iterations=max_iters, progress=False):
    if sampler.iteration==max_iters-1:
        print("The sampler did not converge. Probably either your data is pathological or a linear fit is bad or unconstrained. If you think this is not the case, try increasing the maximum number of iterations, imposing bounds on the parameters, or using a different move method in the sampling.")
        quit()

    if sampler.iteration % 100:
        continue

    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau    



tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

ML = np.max(log_prob_samples_flat)
index = np.where(log_prob_samples_flat==ML)[0][0]
a_ML = samples[index,0]
b_ML = samples[index,1]
s_ML = samples[index,2]

print(a_ML,b_ML,s_ML)