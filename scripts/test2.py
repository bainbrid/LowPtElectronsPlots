# conda create -n tf tensorflow 
# conda create --clone tf --name tf-dev
# conda activate tf-dev
# conda install uproot
# conda install matplotlib
# conda install pillow (PIL not supported anymore, just use pillow as you would PIL, from PIL import ...)

import tensorflow as tf
import numpy as np
import uproot
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
#from IPython import display
import collections 
import awkward

# initialisation

print('version={}, CUDA={}'.format( tf.__version__, 
                                    tf.test.is_built_with_cuda(),
                                    #len(tf.config.list_physical_devices('GPU')) > 0, # GPU={}
                                    ))

base='/Users/bainbrid/Repositories/icenet/standalone'
data_path = base+'/data'
models_path = base+'/models'

# open/parse file 
def load_root_file(filename,nevent=-1) :

    # open file
    file = uproot.open('{}/{}'.format(data_path,filename))
    events = file['ntuplizer']['tree']
    print(type(events))

    # Define VARS
    VARS = [x.decode() for x in events.keys() if b'image_' in x]
    
    # Define X
    X_dict = events.arrays(VARS, namedecode = "utf-8")
    print(type(X_dict))
    print(X_dict)

    quit()

    X = np.array([X_dict[j] for j in VARS])
    X = np.transpose(X)

    # Define Y
    Y = events.array("is_e")

    # PROCESS only nevent
    if nevent >= 0 :
        X = X[0:np.min([X.shape[0],nevent]),:]
        Y = Y[0:np.min([Y.shape[0],nevent])]

    # Indices to filter
    ind = np.ones(X.shape[0], dtype=np.uint8)
    ind = np.logical_and(ind,        X[:, VARS.index('is_egamma')] == False )
    ind = np.logical_and(ind,        X[:, VARS.index('tag_pt')] > 5. )
    ind = np.logical_and(ind, np.abs(X[:, VARS.index('tag_eta')]) < 2.5 )
    ind = np.logical_and(ind,        X[:, VARS.index('has_gsf')] == True )
    ind = np.logical_and(ind,        X[:, VARS.index('gsf_pt')]  > 0.5 )
    ind = np.logical_and(ind, np.abs(X[:, VARS.index('trk_eta')]) < 2.5 )

    # Apply filters
    Y = Y[ind]
    X = X[ind,:]
    
    # return
    return X, Y, VARS

def intensity(energy) : 
    return min(255, int(energy/0.1))
#for energy in np.arange(0.,26.,0.1) : print('energy: {:5.1f}, intensity: {:4.1f}'.format(energy,intensity(energy)))

array_random = (np.random.rand(100, 200)*256).astype(np.uint8)
#array_indexed = numpy.zeros(shape=(100,200)) # (row,col) == (y,x)
array_indexed = np.resize(np.arange(256),20000).reshape(100,200).astype(np.uint8)

img = Image.fromarray(array_indexed)
plt.imshow(np.asarray(img))
plt.savefig(models_path+'/image.pdf')
plt.clf()
plt.close()

#table = awkward.JaggedArray.zip(**tree.arrays(['is_e','image_*']))
#for irow,row in enumerate(table) : print(irow,row)

#print(tf.io.gfile.glob(data_path+'/*')[:10])

def tree_to_protobufs(tree,nevents=-1) :
  events = tree.arrays(['is_e','image_*'], outputtype=collections.namedtuple)
  nevents = min(nevents,len(events.is_e)) if nevents > 0 else len(events.is_e)
  examples = []
  for i in range(nevents) :
    example = tf.train.Example()
    example.features.feature['label'].int64_list.value.append(events.is_e[i])
    example.features.feature['clu_n'].int64_list.value.append(events.image_clu_n[i])
    example.features.feature['clu_eta'].float_list.value.extend(events.image_clu_eta[i])
    example.features.feature['clu_phi'].float_list.value.extend(events.image_clu_phi[i])
    example.features.feature['clu_e'].float_list.value.extend(events.image_clu_e[i])
    examples.append(example)
  return examples

import time
start_time = time.time()
protobufs = tree_to_protobufs(tree,nevents=-1)
print("--- %s seconds ---" % (time.time() - start_time))

print(len(protobufs))
print(protobufs[0])


