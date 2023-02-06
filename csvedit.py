import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('dataset/alljoints3.csv')

df['ShFE_theta'] = df['ShFE_theta']*(-1)
df['ShFE_theta'] = df['ShFE_theta'].sub(df['ShFE_theta'].min())
df['ShAA_theta'] = df['ShAA_theta'].sub(df['ShAA_theta'].min())
df['Elbow_theta'] = df['Elbow_theta'].sub(df['Elbow_theta'].min())

dataset = df.to_numpy(dtype=float)

dir = np.array([],dtype=int).reshape(0,3)
tmpdir = np.array([],dtype=int).reshape(0,3)

dataprev = {'Elbow_tao': 0,
            'ShFE_tao': 0,
            'ShAA_tao': 0}

tdeprev = 0
tdfprev = 0
tdaprev = 0

for index, data in df.iterrows():
    
    if data['Elbow_tao'] > dataprev['Elbow_tao']:
        tde = 0
    elif data['Elbow_tao'] < dataprev['Elbow_tao']:
        tde = 1
    else:
        tde = tdeprev
    dataprev['Elbow_tao'] = data['Elbow_tao']
    tdeprev = tde
    
    
    if data['ShFE_tao'] > dataprev['ShFE_tao']:
        tdf = 0
    elif data['ShFE_tao'] < dataprev['ShFE_tao']:
        tdf = 1
    else:
        tdf = tdfprev
    dataprev['ShFE_tao'] = data['ShFE_tao']
    tdfprev = tdf
    
    
    if data['ShAA_tao'] > dataprev['ShAA_tao']:
        tda = 0
    elif data['ShAA_tao'] < dataprev['ShAA_tao']:
        tda = 1
    else:
        tda = tdaprev
    dataprev['ShAA_tao'] = data['ShAA_tao']
    tdaprev = tda
    
    
    tmpdir = np.append(tmpdir,[[tde, tdf, tda]], axis=0)
    if index%1000 ==999:
        # calling for memory in batches to make it faster
        dir = np.concatenate((dir,tmpdir),axis=0)
        # np.delete(tmpdir,[0,1,2],axis=1)
        tmpdir = np.array([],dtype=int).reshape(0,3)
        
        print(index/1000)
dir = np.concatenate((dir,tmpdir),axis=0)

newdataset = np.concatenate((dataset[:,0:3],dir), axis=1)
newdataset = np.concatenate((newdataset,dataset[:,3:6]), axis=1)

np.savetxt("dataset/alljoints3_labeled.csv", newdataset, delimiter=",")




