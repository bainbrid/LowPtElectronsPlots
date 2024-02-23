import numpy as np
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
from cmsjson import CMSJson
from pdb import set_trace
import os

parser = ArgumentParser()
parser.add_argument('model')
parser.add_argument('--dataset')
parser.add_argument('--what')

parser.add_argument("--gpu",  help="select specific GPU",   type=int, metavar="OPT", default=-1)
parser.add_argument("--gpufraction",  help="select memory fraction for GPU",   type=float, metavar="OPT", default=0.5)

args = parser.parse_args()

import pandas as pd
import json
import matplotlib.pyplot as plt
from glob import glob
if args.model.endswith('.csv'):
    #TODO, make the BO part
    bo = pd.read_csv(args.model)
    best = bo.target.argmax()
    pars = dict(bo.loc[best])
    del pars['target']
    base = os.path.dirname(args.model)
    #unfortunately the hash does not mean anything :(
    for jfile in glob('%s/train_bo_*/hyperparameters.json' % base):
        #check for ~equality
        jpars = json.load(open(jfile))
        if not set(jpars.keys()) == set(pars.keys()):
            print 'skipping', jfile
            continue
        equals = all(
            abs(pars[i] - jpars[i])/(abs(pars[i]) if pars[i] else 1) < 10**-3 
            for i in pars
            )
        #except: set_trace()
        if equals:
            break
    else: #for else! like, the third time I use it!
        raise RuntimeError('I cannot find the training dir')
    
    train_dir = os.path.dirname(jfile)
    model = '%s/KERAS_check_best_model.h5' % train_dir
    dataset = glob('%s/*.hdf' % base)[0]
    args.what = base.split('nn_bo_')[1].replace('_noweight','')
    plots = base
else:
    model = args.model
    dataset = args.dataset
    plots = os.dirname(model)
    if not dataset:
        raise RuntimeError('You must specify a dataset if you are not running in Bayesian Optimization mode')

cmssw_path = os.path.dirname(os.path.realpath(__file__)).split('src/LowPtElectrons')[0]
os.environ['CMSSW_BASE'] = cmssw_path


from keras import backend as K, callbacks
from keras.models import load_model
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

#this should be outsorced
from features import *
if args.what == 'seeding':
   features = seed_features
   additional = seed_additional
elif args.what == 'fullseeding':
   features = fullseed_features
   additional = seed_additional
elif args.what == 'id':
   features = id_features
   additional = id_additional+['gsf_ecal_cluster_ematrix']
else:
   raise ValueError()

model = load_model(model)
test = pd.read_hdf(dataset, key='data')

#
# plot performance
#
from sklearn.metrics import roc_curve, roc_auc_score
#this should go in the outsourced part as well!
if any('crystal' in i for i in test.columns):
    ncrystals = len([i for i in test.columns if 'crystal' in i])
    new_features = ['crystal_%d' % i for i in range(ncrystals)]
    features += new_features
training_out = model.predict(test[features].as_matrix())
roc = roc_curve(
   test.is_e.as_matrix().astype(int), 
   training_out)[:2]
auc_score = roc_auc_score(test.is_e, training_out)

# make plots
plt.figure(figsize=[8, 8])
plt.title('%s training' % args.what)
plt.plot(
   np.arange(0,1,0.01),
   np.arange(0,1,0.01),
   'k--')
plt.plot(*roc, label='Retraining (AUC: %.2f)'  % auc_score)
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
plt.savefig('%s/test_NN.png' % (plots))
plt.savefig('%s/test_NN.pdf' % (plots))
plt.gca().set_xscale('log')
plt.xlim(1e-4, 1)
plt.savefig('%s/test_log_NN.png' % (plots))
plt.savefig('%s/test_log_NN.pdf' % (plots))
plt.clf()
