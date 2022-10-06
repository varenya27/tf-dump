import numpy as np
c = 5
for i in range(1,7):
    fp = open('/values'+str(i)+'.txt','w')
    with open('raw_data.txt ') as f:
        for line in f:
            line = line.split()
            line1=line[1]+' '+line[2]+' '
            # line1=str(float(line[1])*2.303)+' '+str(float(line[2])*2.303)+' '
            if(line[c]!='0.0'):
                line1+= str(np.log10(float(line[c])) )+' '
                line1+=str ( (float(line[c+1])/float(line[c]))/np.log(10) )
                fp.write(line1+'\n')
    c+=2
    fp.close()