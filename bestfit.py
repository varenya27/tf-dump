import numpy as np 
from matplotlib import pyplot as plt 

m,c=[],[]
vel=['$V_f$','$V_{2.2}$','$V_{Re}$','$V_{max}$','$W_{P20}/2$','$W_{M50}/2$',]
with open('bestfit.txt','r') as f:
    for line in f:
        line=line.split()
        m.append(float(line[0]))
        c.append(float(line[3]))
for i in range(6):
    y, err_y, x, err_x= np.loadtxt("values"+str(i+1)+".txt", unpack=True)
    plt.figure()
    plt.errorbar(x,y,err_y,err_x,ls='none')
    xp = np.linspace(min(x)*0.9,max(x)*1.1)
    plt.plot(xp,m[i]*xp+c[i])
    plt.grid()
    plt.title('Best Fit Plot for: '+vel[i]+' : m = '+str(m[i])+', c = '+str(c[i]))
    plt.xlabel ('$log (v)kms^{-1}$')
    plt.ylabel ('$log (M_{bar}/M_\odot)$')
    # plt.show()
    plt.savefig('bestfit_'+str(i+1)+'.png')
    