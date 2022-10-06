data=[]
with open('codes/raw_data.txt','r') as f:
    i=0
    for line in f:
        i+=1
        if(i<31):
            continue
        line=line.split()
        data.append( [line[0],line[1],line[2],line[5],line[6]])
with open('codes/data.txt','w') as f:
    for galaxy in data:
        f.write(' '.join(map(str, galaxy))+'\n')