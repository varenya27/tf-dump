# !/usr/bin/env python
import numpy as np
import math
import BayesLineFit_mod as blf

# ###############################################################################
#
# Copyright (C) 2018-2020, Harry Desmond & Federico Lelli
# E-mail: harry.desmond@physics.ox.ac.uk, LelliF@cardiff.ac.uk
#
# Up-to-date versions of the software are available from:
# https://www2.physics.ox.ac.uk/contacts/people/desmond
# http://astroweb.cwru.edu/SPARC/
# https://www.lellifederico.com/
#
# If you have used this software in your research, please acknowledge
# Lelli et al. (2019, MNRAS, 484, 2367) where we describe in detail 
# the maximum-likelihood (ML) method to fit a linear model y=a*x+b
# considering either vertical or the orthogonal intrinsic scatter.
#
# This software is provided as it is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
#################################################################################
#
# USAGE: 
#  In interactive mode, the user can simply run "BayesLineFit.py" with python-3
#  and follow the instructions. In batch/scripting mode, the user can call the
#  function BayesLineFit_mod.BayesLineFit as illustrated in the examples below.
#         
#################################################################################
if __name__ == "__main__":
    #import data from "example.dat" file
    if(1):
        y, err_y, x, err_x= np.loadtxt("Tully-Fisher\BayesLineFit\Ve.txt", unpack=True)

        #EXAMPLE 1: Run BayesLineFit with vertical intrinsic scatter and default options.
        #           Several output text files and plots are produced.
        a, b, s, sobs = blf.BayesLineFit(x, y, err_x, err_y, orthfit=False,max_iters=15000)

        #EXAMPLE 2: Run BayesLineFit with orthogonal intrinsic scatter changing default options. 
        #           Plots and screen outputs are suppressed & the MCMC is run longer with more walkers.
        # a, b, s, sobs = blf.BayesLineFit(x, y, err_x, err_y, nwalkers=100, max_iters=15000, quiet=True,
        #                                 outfile_chain=None, outfile_bestfit=None, outplot_convergence=None,
        #                                 outplot_corner=None, outplot_bestfit=None)
        print(a,b,s,sobs)
    else:
        for i in range(1,7):
            y, err_y, x, err_x= np.loadtxt("BayesLineFit/values"+str(i)+".txt", unpack=True)

            #EXAMPLE 1: Run BayesLineFit with vertical intrinsic scatter and default options.
            #           Several output text files and plots are produced.
            a, b, s, sobs = blf.BayesLineFit(x, y, err_x, err_y, orthfit=False)

            #EXAMPLE 2: Run BayesLineFit with orthogonal intrinsic scatter changing default options. 
            #           Plots and screen outputs are suppressed & the MCMC is run longer with more walkers.
            # a, b, s, sobs = blf.BayesLineFit(x, y, err_x, err_y, nwalkers=100, max_iters=15000, quiet=True,
            #                                 outfile_chain=None, outfile_bestfit=None, outplot_convergence=None,
            #                                 outplot_corner=None, outplot_bestfit=None)
            print(a,b,s,sobs)
            fp = open('final.txt','a')
            fp.write(str(a[0])+' ' + str(b[0])+' ' +str(s[0])+'\n')
            fp.write("Slope (ML, median, upper error, lower error): %s; %s; +%s, %s\n" % (float('%.4g'%a[0]), float('%.4g'%a[1]), float('%.4g'%a[2]), float('%.4g'%a[3])))
            fp.write("Intercept (ML, median, upper error, lower error): %s; %s; +%s; %s\n" % (float('%.4g'%b[0]), float('%.4g'%b[1]), float('%.4g'%b[2]), float('%.4g'%b[3])))
            fp.write("Intrinsic scatter (ML, median, upper error, lower error): %s; %s; +%s; %s\n" % (float('%.4g'%s[0]), float('%.4g'%s[1]), float('%.4g'%s[2]), float('%.4g'%s[3])))
            fp.write("***\n")
            fp.close()
    # See docstring for detailed API
