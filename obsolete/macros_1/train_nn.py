import numpy as np
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
from cmsjson import CMSJson
from pdb import set_trace

parser = ArgumentParser()
parser.add_argument('what')
parser.add_argument(
   '--jobtag', default='', type=str
)

parser.add_argument("--gpu",  help="select specific GPU",   type=int, metavar="OPT", default=-1)
parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=0.17)

args = parser.parse_args()
#dataset = 'all' 

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
import os
dataset = target_dataset

ccmssw_path = dir_path = os.path.dirname(os.path.realpath(__file__)).split('src/LowPtElectrons')[0]
os.environ['CMSSW_BASE'] = cmssw_path
mods = '%s/src/LowPtElectrons/LowPtElectrons/macros/models/%s/' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(mods):
   os.makedirs(mods)

if args.jobtag:
    jobdir = '%s/%s' % (mods, args.jobtag)
    if not os.path.isdir(jobdir):
        os.makedirs(jobdir)
else:
    jobdir = mods

plots = '%s/src/LowPtElectrons/LowPtElectrons/macros/plots/%s/' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(plots):
   os.makedirs(plots)

from features import *
features, additional = get_features(args.what)

fields = features+labeling+additional
if 'gsf_pt' not in fields : fields += ['gsf_pt']
data = pre_process_data(dataset, fields, args.what in ['seeding', 'fullseeding'])

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)


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

# These should become arguments for the bayesian optimization
n_layers = 3
n_nodes = 2*len(features)
dropout = 0.1
learn_rate = 0.001
batch_size = 400
n_epochs = 70

# Make model
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

model.compile(
   loss = 'binary_crossentropy',
   optimizer=Adam(lr=0.0000001), 
   metrics = ['binary_accuracy']
   )

print model.summary()

#train batch norm
model.fit(
   train[features].as_matrix(),
   train.is_e.as_matrix().astype(int),
   sample_weight=train.weight.as_matrix(),
   batch_size=batch_size, epochs=1, verbose=1, validation_split=0.2,
)

from DeepJetCore.modeltools import set_trainable
#fix batch norm to get the total means and std_dev 
#instead of the batch one
model = set_trainable(model, 'globals_input_batchnorm', False)
model.compile(
   loss = 'binary_crossentropy',
   optimizer=Adam(lr=learn_rate),
   metrics = ['binary_accuracy']
   )

from DeepJetCore.training.DeepJet_callbacks import DeepJet_callbacks
callbacks = DeepJet_callbacks(
   model, 
   outputDir=jobdir,
   stop_patience=15,
   lr_patience = 7,   
)

#train batch norm
hh = model.fit(
   train[features].as_matrix(),
   train.is_e.as_matrix().astype(int),
   sample_weight=train.weight.as_matrix(),
   batch_size=batch_size, epochs=n_epochs, verbose=1, validation_split=0.2,
   callbacks=callbacks.callbacks
)
set_trace()



#
# plot performance
#
from sklearn.metrics import roc_curve, roc_auc_score
args_dict = args.__dict__

rocs = {}
for df, name in [
   (train, 'train'),
   (test, 'test'),
   ]:
   training_out = model.predict(df[features].as_matrix())
   rocs[name] = roc_curve(
      df.is_e.as_matrix().astype(int), 
      training_out)[:2]
   args_dict['%s_AUC' % name] = roc_auc_score(df.is_e, training_out)

# make plots
plt.figure(figsize=[8, 8])
plt.title('%s training' % args.what)
plt.plot(
   np.arange(0,1,0.01),
   np.arange(0,1,0.01),
   'k--')
plt.plot(*rocs['test'], label='Retraining (AUC: %.2f)'  % args_dict['test_AUC'])
if args.what in ['seeding', 'fullseeding']:
   eff = float((data.baseline & data.is_e).sum())/data.is_e.sum()
   mistag = float((data.baseline & np.invert(data.is_e)).sum())/np.invert(data.is_e).sum()
   plt.plot([mistag], [eff], 'o', label='baseline', markersize=5)
elif args.what == 'id':
   mva_v1 = roc_curve(test.is_e, test.ele_mvaIdV1)[:2]   
   mva_v2 = roc_curve(test.is_e, test.ele_mvaIdV2)[:2]
   mva_v1_auc = roc_auc_score(test.is_e, test.ele_mvaIdV1)
   mva_v2_auc = roc_auc_score(test.is_e, test.ele_mvaIdV2)
   plt.plot(*mva_v1, label='MVA ID V1 (AUC: %.2f)'  % mva_v1_auc)
   plt.plot(*mva_v2, label='MVA ID V2 (AUC: %.2f)'  % mva_v2_auc)
else:
   raise ValueError()

plt.xlabel('Mistag Rate')
plt.ylabel('Efficiency')
plt.legend(loc='best')
plt.xlim(0., 1)
plt.savefig('%s/%s_%s_%s_NN.png' % (plots, dataset, args.jobtag, args.what))
plt.savefig('%s/%s_%s_%s_NN.pdf' % (plots, dataset, args.jobtag, args.what))
plt.gca().set_xscale('log')
plt.xlim(1e-4, 1)
plt.savefig('%s/%s_%s_%s_log_NN.png' % (plots, dataset, args.jobtag, args.what))
plt.savefig('%s/%s_%s_%s_log_NN.pdf' % (plots, dataset, args.jobtag, args.what))
plt.clf()
