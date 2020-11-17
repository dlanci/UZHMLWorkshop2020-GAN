
import numpy as np
import os 
import pickle
import matplotlib.pyplot as plt

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


def generate_and_save_images(model, epoch, test_input, maxval, save_img=False, path=None):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(16,16))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * (maxval))
        plt.colorbar()
        plt.axis('off')

    if save_img and path is not None:
        os.makedirs(path, exist_ok=True)
        file_path = os.join(path, 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.savefig(file_path)
    plt.show()


def generate_and_save_images_conditional(model, epoch, test_input, X_test, label_test, maxval, save_img=False, path=None):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model([test_input, label_test], training=False)
    #print(labels_input)
    X_test_np = X_test.numpy()
    
    fig = plt.figure(figsize=(16,16))

    for i in range(8):
        plt.subplot(4, 4, 2*i+1 )
        plt.imshow(predictions[i, :, :, 0] * (maxval) )
        plt.colorbar()
        plt.subplot(4, 4, 2*i+2 )
        plt.imshow(X_test_np[i, :, :, 0] * (maxval) )
        plt.colorbar()
        plt.axis('off')


    if save_img and path is not None:
        os.makedirs(path, exist_ok=True)
        file_path = os.join(path, 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.savefig(file_path)
        
    plt.show()