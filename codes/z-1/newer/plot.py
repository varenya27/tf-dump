import emcee
import numpy as np
from scipy.odr import Model, RealData, ODR
import matplotlib as mpl
from matplotlib import pyplot as plt# import pyplot from matplotlib
from time import time               # use for timing functions
import corner


# v_=['new/Ve','new/Vopt','original/Vout','new/Ve','original/Vopt','original/Vout','original/Ve_st','original/Vopt_st','original/Vout_st']
# v='new/Vout_swapped'
# v='values1'
v='newer/Ve'
y, err_y, x, err_x= np.loadtxt(v+".txt", unpack=True)
# print(y,x)

plt.figure()
plt.grid()
# plt.scatter(x,y,color='black',s=1)
plt.xlim(1.4,2.8)
plt.ylim(8,12)
plt.errorbar(x, y,err_y,err_x,ls='none')
print(min(x),max(x))
plt.xlabel ('$log v$')
plt.ylabel ('$log M_{bar}$')
plt.title('Data with the error bars for: '+v)
# plt.savefig('data_'+v+'.png')
plt.show()