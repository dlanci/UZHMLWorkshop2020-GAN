
import numpy as np
import os 
import pickle

def load_dataset(path='/disk/lhcb_data/davide/ML_UZH_ds/', test=False):
    
    ds=[]
    tot_evts=0

    max_batches = 1 if test else 5 

    for i in range(max_batches):
    
        filename = os.path.join(path,'batch{0}.pickle'.format(i))
        print("Opening file: ",filename)
        with open(filename, 'rb') as f:
            cd=pickle.load(f)
            size=cd[list(cd.keys())[0]].shape[0]
            ds.append(cd)
            tot_evts+=size
            print("Dataset {0} contains {1} events".format(i, size))  
            
    print("Total train set contains {0} events".format(tot_evts))
    tuple_={}        
    for k in ds[0].keys():
        tuple_[k]=np.concatenate(list(d[k] for d in ds))
    
    return tuple_, tot_evts


