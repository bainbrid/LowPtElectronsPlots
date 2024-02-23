import numpy as np
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
from cmsjson import CMSJson
from pdb import set_trace
import os

parser = ArgumentParser()
parser.add_argument('what')

parser.add_argument(
   '--jobtag', default='', type=str
)
parser.add_argument(
   '--test', action='store_true'
)
parser.add_argument(
   '--noweight', action='store_true'
)
parser.add_argument(
   '--recover', action='store_true', 
   help='recover lost best iteration due to bug'
)

parser.add_argument("--gpu",  help="select specific GPU",   type=int, metavar="OPT", default=-1)
parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=0.5)

args = parser.parse_args()
#dataset = 'test'

cmssw_path = dir_path = os.path.dirname(os.path.realpath(__file__)).split('src/LowPtElectrons')[0]
os.environ['CMSSW_BASE'] = cmssw_path

from keras import backend as K, callbacks
import tensorflow as tf
if args.gpu<0:
    import imp
    try:
        imp.find_module('setGPU')
        import setGPU
    except ImportError:
        found = False
else:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('running on GPU '+str(args.gpu))

if args.gpufraction>0 and args.gpufraction<1:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpufraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)
    print('using gpu memory fraction: '+str(args.gpufraction))


import matplotlib.pyplot as plt
import ROOT
import uproot
#import rootpy
import json
import pandas as pd
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
from datasets import tag, pre_process_data, target_dataset
dataset = 'test' if args.test else target_dataset

plots = '%s/src/LowPtElectrons/LowPtElectrons/macros/plots/%s/' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(plots):
   os.makedirs(plots)

mods = '%s/src/LowPtElectrons/LowPtElectrons/macros/models/%s/' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(mods):
   os.makedirs(mods)

opti_dir = '%s/nn_bo_%s' % (mods, args.what)
if args.noweight:
   opti_dir += '_noweight'
if not os.path.isdir(opti_dir):
   os.makedirs(opti_dir)

from features import *
features, additional = get_features(args.what)

fields = features+labeling+additional
if 'gsf_pt' not in fields : fields += ['gsf_pt']
data = pre_process_data(dataset, fields, args.what in ['seeding', 'fullseeding'])
if args.noweight:
   data.weight = 1

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)
test.to_hdf(
   '%s/nn_bo_%s_testdata.hdf' % (opti_dir, args.what),
   'data'
   ) 

train, validation = train_test_split(train, test_size=0.2, random_state=42)

#
# Train NN
#

from keras.models import Model
from keras.layers import Input
from keras.metrics import binary_accuracy
from keras.initializers import RandomNormal
from keras.layers import Dense, Dropout, Multiply, Add, \
    Concatenate, Reshape, LocallyConnected1D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from sklearn.metrics import roc_curve, roc_auc_score

# These should become arguments for the bayesian optimization

# Make model
def make_model(n_layers = 3, n_nodes = 2*len(features), dropout = 0.1):
    #sanitize inputs
    n_layers = int(n_layers)
    n_nodes = int(n_nodes)    
    inputs = [Input((len(features),))]
    normed = BatchNormalization(
        momentum=0.6,
        name='globals_input_batchnorm')(inputs[0])
    
    layer = normed
    for idx in range(n_layers):
        layer = Dense(n_nodes, activation='relu', 
                      name='dense_%d' % idx)(layer)
        layer = Dropout(dropout)(layer)
        
    output = Dense(1, activation='sigmoid', name='output')(layer)
    model = Model(
        inputs=inputs, 
        outputs=[output], 
        name="eid"
        )
    return model

from DeepJetCore.modeltools import set_trainable
from DeepJetCore.training.DeepJet_callbacks import DeepJet_callbacks
from keras.callbacks import Callback, EarlyStopping, History, ModelCheckpoint
from DeepJetCore.training.ReduceLROnPlateau import ReduceLROnPlateau
from keras.models import load_model
iteration_idx = 0

def train_model(**kwargs):
    print 'training:', kwargs
    global iteration_idx
    train_dir = '%s/train_bo_%d' % (opti_dir, iteration_idx)
    iteration_idx += 1
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
        
    learn_rate = 10.**kwargs['log_learn_rate']
    batch_size = int(kwargs['batch_size'])
    n_epochs   = 1000 #int(kwargs['n_epochs'])

    del kwargs['log_learn_rate']
    del kwargs['batch_size']
    # del kwargs['n_epochs']  
    
    model = make_model(**kwargs)

    model.compile(
        loss = 'binary_crossentropy',
        optimizer=Adam(lr=0.0000001), 
        metrics = ['binary_accuracy']
        )
    #train batch norm
    model.fit(
        train[features].as_matrix(),
        train.is_e.as_matrix().astype(int),
        sample_weight=train.weight.as_matrix(),
        batch_size=batch_size, epochs=1, verbose=0, 
        validation_data=(
            validation[features].as_matrix(), 
            validation.is_e.as_matrix().astype(int),
            validation.weight.as_matrix()
            )
        )
    
    #fix batch norm to get the total means and std_dev 
    #instead of the batch one
    model = set_trainable(model, 'globals_input_batchnorm', False)
    model.compile(
        loss = 'binary_crossentropy',
        optimizer=Adam(lr=learn_rate),
        metrics = ['binary_accuracy']
        )

    print model.summary()
    callbacks = DeepJet_callbacks(
        model, 
        outputDir=train_dir,
        stop_patience=999,
        lr_patience = 10,
        verbose=False
        )

    history = model.fit(
        train[features].as_matrix(),
        train.is_e.as_matrix().astype(int),
        sample_weight=train.weight.as_matrix(),
        batch_size=batch_size, epochs=n_epochs, 
        verbose=2, callbacks=callbacks.callbacks,
        validation_data=(
            validation[features].as_matrix(), 
            validation.is_e.as_matrix().astype(int),
            validation.weight.as_matrix()
            )
        )
    
    clf = load_model(train_dir+"/KERAS_check_best_model.h5")
    training_out = clf.predict(validation[features].as_matrix())
    training_out[np.isnan(training_out)] = -999 #happens rarely, but happens
    auc = roc_auc_score(validation.is_e, training_out)

    kwargs['nepochs'] = len(history.history['loss'])
    with open('%s/hyperparameters.json' % train_dir, 'w') as j:
        j.write(json.dumps(kwargs))

    return auc

#
# Bayesian optimization
#
from xgbo import BayesianOptimization
par_space = {
    'n_layers'   : (2, 10), 
    'n_nodes'    : (len(features)/3, 3*len(features)), 
    'dropout'    : (0., 0.8),
    'log_learn_rate' : (-4., -1),
    'batch_size' : (500, 2000),
    #'n_epochs'   : (100, 500),
    }

bo = BayesianOptimization(
    train_model,
    par_space,
    verbose=1,
    checkpoints='%s/checkpoints.csv' % opti_dir
)
bo.init(5, sampling='lhs')
bo.maximize(5, 50)

bo.print_summary()
with open('%s/nn_bo.json' % opti_dir, 'w') as j:
    mpoint = bo.space.max_point()
    thash = mpoint['max_params'].__repr__().__hash__()
    info = mpoint['max_params']
    info['value'] = mpoint['max_val']
    info['hash'] = thash
    j.write(json.dumps(info))

if args.recover:
   ## best_id = bo.Y.argmax()
   ## iteration_idx = best_id
   ## best_pars = bo.space.max_point()['max_params']
    iteration_idx = 666
    train_model(n_layers=8, n_nodes=2*len(features), dropout=0.1, log_learn_rate=-3, batch_size=700)

