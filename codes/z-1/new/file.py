import csv
import numpy as np
Ve,Ve_err,Vopt,Vopt_err,Vout,Vout_err,M,M_err=[],[],[],[],[],[],[],[]
file='new\GS21_catalog_P3.csv'
k=0
with open(file,'r') as f:
    reader=csv.reader(f)
    for line in reader:
        # if(line[22]=='F'): continue
        if k==0: 
            k=1
            continue
        ve=float(line[11])
        Ve.append(np.log10(ve))
        ve_err=float(line[12])
        Ve_err.append(ve_err/(ve*2.303))
        vopt=float(line[13])
        Vopt.append(np.log10(vopt))
        vopt_err=float(line[14])
        Vopt_err.append(vopt_err/(vopt*2.303))       
        vout = float(line[15])
        Vout.append(np.log10(vout))
        vout_err=float(line[16])
        Vout_err.append(vout_err/(2.303*vout))
        m=(float(line[17])+float(line[20]))
        M.append(np.log10(m))
        m_err=np.sqrt(float(line[18])**2+float(line[19])**2)
        m_err=m_err/(2.303*m)
        M_err.append(m_err)
print(max(Ve),min(Ve))
print(max(M),min(M))
Ve_file='Ve_swapped.txt'
Vopt_file='Vopt_swapped.txt'
Vout_file='Vout_swapped.txt'
with open(Ve_file,'w') as f:
    for i in range(len(Ve)):
        x=str(M[i])+' '+str(M_err[i])+' '+str(Ve[i])+' '+str(Ve_err[i])+'\n'
        f.write(x)
with open(Vopt_file,'w') as f:
    for i in range(len(Ve)):
        x=str(M[i])+' '+str(M_err[i])+' '+str(Vopt[i])+' '+str(Vopt_err[i])+'\n'
        f.write(x)
with open(Vout_file,'w') as f:
    for i in range(len(Ve)):
        x=str(M[i])+' '+str(M_err[i])+' '+str(Vout[i])+' '+str(Vout_err[i])+'\n'
        f.write(x)