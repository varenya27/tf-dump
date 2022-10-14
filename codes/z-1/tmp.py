import emcee
import numpy as np
from scipy.odr import Model, RealData, ODR
import matplotlib as mpl
from matplotlib import pyplot as plt# import pyplot from matplotlib
import time               # use for timing functions
import corner

def straight_line(theta,x):
    y=theta[0]*x + theta[1]
    return y

min_ = 0
max_ = 5
min_scat = -5.
max_scat = 0.

def logprior(theta):
    lp = 0.
    m, c = theta
    # m, c = theta


    # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
    # (we don't care about it being properly normalised, but you can if you want) 
    lp = 0. if min_ < c < max_ and min_ < m < max_  else -np.inf

    # Gaussian prior on m
    mmu = 0.     # mean of the Gaussian prior
    msigma = 10. # standard deviation of the Gaussian prior
    lp -= 0.5 * ((m - mmu) / msigma)**2

    return lp


def logchi(theta, y, x, err_y):
    m,c=theta
    expected = m*x+c
    delta = y - expected
    chi_sq = np.sum(delta**2/err_y**2)
    # print(chi_sq)
    return -0.5*(chi_sq)

def logposterior(theta, y, x, err_y, err_x):
    lp = logprior(theta) # get the prior
    # if the prior is not finite return a probability of zero (log probability of -inf)
    if not np.isfinite(lp):
        return -np.inf

    # return the likeihood times the prior (log likelihood plus the log prior)
    # return lp + lnprob_vertical(theta, x, err_x, y, err_y)
    # return lp + loglikelihood(theta, y, x, err_y, err_x)
    return lp + logchi(theta, y, x, err_y)



Nens = 300 #300 # number of ensemble points
ndims = 2
Nburnin = 500  #500 # number of burn-in samples
Nsamples = 500  #500 # number of final posterior samples
# v_=['Ve','Vopt','Vout','Ve','Vopt','Vout','Ve_st','Vopt_st','Vout_st']
# v=v_[-3]
velocities=['Vout','Vout_normalized','Vout_st','Vout_st_normalized']
for v in velocities:
    y, err_y, x, err_x= np.loadtxt('newer/'+v+".txt", unpack=True)
    argslist = (y, x, err_y, err_x)



    p0 = []
    for i in range(Nens):
        pi = [
            np.random.uniform(min_,max_), 
            np.random.uniform(min_,max_),
        ]
        # pi = [np.random.uniform(2, 4), np.random.uniform(1, 3), np.random.uniform(rms_ODR/10., rms_ODR)]
        # pi = [np.random.normal(a_ODR, err_a_ODR), np.random.normal(b_ODR, err_b_ODR), np.random.uniform(rms_ODR/10., rms_ODR)]
        p0.append(pi)

    # set up the sampler    
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)
    # pass the initial samples and total number of samples required
    t0 = time.time() # start time
    sampler.run_mcmc(p0, Nsamples + Nburnin);
    t1 = time.time()

    timeemcee = (t1-t0)
    print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

    # extract the samples (removing the burn-in)
    samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)

    #plots
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    m_final = np.percentile(flat_samples[:, 0], [16, 50, 84])[1]
    c_final = np.percentile(flat_samples[:, 1], [16, 50, 84])[1]
    Med_value = [m_final,c_final]

    figure = corner.corner(
        flat_samples,
        # title_fmt=".2E",
        levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), 
        # levels=(0.68,0.95,0.99), 
        labels=[r"Slope", r"Intercept"], 
        quantiles=[0.16,0.84], 
        range=[(m_final-m_final/100,m_final+m_final/100), (c_final-c_final/100,c_final+c_final/100)],
        show_titles=True, 
        label_kwargs={"fontsize": 12},
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
    figure.savefig('chi_corner_'+v+'.png',format='png', dpi=300)
    # figure = corner.corner(samples, levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), labels=[r"Slope", r"Intercept", r"Intrinsic Scatter", r"Intrinsic Scatter"], quantiles=[0.16,0.84], show_titles=True, label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 10}, range=[(a_ML-0.4,a_ML+0.4), (b_ML-0.6,b_ML+0.6), (s_ML-0.1,s_ML+0.1)])

    line=[]
    results = '\n'+v+' at time '+(time.asctime( time.localtime(time.time()) )[11:19])+'\n'+'\n' + v
    labels=['slope = ','intercept = ','intrinsic scatter = ',]
    for i in range(ndims):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        # print(round(mcmc[1],3), round(q[0],3), round(-q[1],3))
        print(labels[i],round(mcmc[1],3), round(q[0],4), round(-q[1],4))
        results+='&$'+str(round(mcmc[1],3))+ '^{+'+str(round(q[0],4))+'}_{'+ str(round(-q[1],4))+'}$&'
        line.append(mcmc[1])
        # display(Math(txt))
    with open('results.txt','a') as f:
        f.write(results)
    plt.figure()
    plt.grid()
    plt.scatter(x,y,color='black',s=1)
    # plt.xlim(0,5)
    # plt.ylim(5,15)
    plt.errorbar(x, y,err_y,err_x,ls='none')

    x_line=np.linspace(min(x)-min(x)/10,max(x)+max(x)/10)

    plt.plot(x_line, line[0]*x_line+line[1],color='black')
    plt.xlabel ('$log v$')
    # plt.ylabel ('$log M_{bar}$')
    ylabel ='$log M_{star}$' if ('st' in v)  else '$log M_{bar}$'
    plt.ylabel(ylabel)
    plt.title(v)
    plt.savefig('chi_bestfit_'+v+'.png')
    # plt.show()