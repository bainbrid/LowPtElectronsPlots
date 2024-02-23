################################################################################
# libraries 

import numpy as np
import pandas as pd

import ROOT
import uproot
import rootpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("cms")

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
#rc('text.latex', preamble='\usepackage{sfmath}')

from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties

from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
from xgbo.xgboost2tmva import convert_model
from itertools import cycle

import json
import os

from plotting import *

################################################################################
print "### Command line args ..."

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--what',default='cmssw_mva_id_extended',type=str)
parser.add_argument('--tag',default='2019Oct10',type=str)
parser.add_argument('--dataset',default='test',type=str)
parser.add_argument('--nevents',default=-1,type=int)
#
parser.add_argument('--train',action='store_true')
parser.add_argument('--config',default='models/rob.json',type=str)
parser.add_argument('--nthreads', default=8, type=int)
parser.add_argument('--no_early_stop', action='store_true')
parser.add_argument('--xml', action='store_true')
#
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--jobtag',default='',type=str) # obsolete?
args = parser.parse_args()
print "Command line args:",vars(args)

################################################################################
print "### Define parameters ..."

input_files = {
   'test':['/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15/src/2-ntuples-from-crab/output.LATEST.root']#output.CURRENT.root
}

cmssw_base='/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15'
base='%s/src/LowPtElectrons/LowPtElectrons/macros'%(cmssw_base)
mods='%s/models/%s'%(base,args.tag)
mods_='%s/bdt_%s'%(mods,args.what)
plots='%s/plots/%s'%(base,args.tag)
if not os.path.isdir(mods) : os.makedirs(mods)
if not os.path.isdir(mods_): os.makedirs(mods_)
if not os.path.isdir(plots) : os.makedirs(plots)

# cmssw_mva_id
features = [ 
   'eid_rho',
   'eid_ele_pt',
   'eid_sc_eta',
   'eid_shape_full5x5_sigmaIetaIeta',
   'eid_shape_full5x5_sigmaIphiIphi',
   'eid_shape_full5x5_circularity',
   'eid_shape_full5x5_r9',
   'eid_sc_etaWidth',
   'eid_sc_phiWidth',
   'eid_shape_full5x5_HoverE',
   'eid_trk_nhits',
   'eid_trk_chi2red',
   'eid_gsf_chi2red',
   'eid_brem_frac',
   'eid_gsf_nhits',
   'eid_match_SC_EoverP',
   'eid_match_eclu_EoverP',
   'eid_match_SC_dEta',
   'eid_match_SC_dPhi',
   'eid_match_seed_dEta',
   'eid_sc_E',
   'eid_trk_p',
]
if 'extended' in args.what : features += ['gsf_bdtout1']

additional = [
   'gen_pt','gen_eta', 
   'trk_pt','trk_eta','trk_charge','trk_dr',
   'seed_trk_driven','seed_ecal_driven',
   'gsf_pt','gsf_eta','gsf_dr','gsf_bdtout2',
   'pfgsf_pt','pfgsf_eta',
   'ele_pt','ele_eta','ele_dr','ele_mva_value','ele_mva_id',
   'evt','weight'
]

labelling = [
   'is_e','is_egamma',
   'has_trk','has_seed','has_gsf','has_pfgsf','has_ele',
   'seed_trk_driven','seed_ecal_driven'
]

fields = features + additional + labelling
fields = list(set(fields))

################################################################################
print "### Load files ..."

if args.dataset not in input_files:
   raise ValueError('The dataset %s does not exist, I have %s' % (args.dataset, ', '.join(input_files.keys())))

print 'getting files from "%s": ' % args.dataset
print ' \n'.join(input_files[args.dataset])
infiles = [ uproot.open(i) for i in input_files[args.dataset] ]

path='ntuplizer/tree' # 'features/tree'
print 'available branches:\n',infiles[0][path].keys()
try:
   ret = infiles[0][path].arrays(fields)
except KeyError as ex:
   print 'Exception! ', ex
   set_trace()
   raise RuntimeError('Failed to open %s properly' % infiles[0])
for infile in infiles[1:]:
   try:
      arrays = infile[path].arrays(fields)
   except:
      raise RuntimeError('Failed to open %s properly' % infile)
   for column in fields:
      ret[column] = np.concatenate((ret[column],arrays[column]))
data_dict = ret

################################################################################
print "### Pre-process data ..."

data = pd.DataFrame(data_dict)
if args.nevents > 0 : data = data.head(args.nevents)

log_trkpt = np.log10(data.trk_pt)
log_trkpt[np.isnan(log_trkpt)] = -9999
data['log_trkpt'] = log_trkpt
data['prescale'] = data.weight
data['weight'] = np.ones(data.weight.shape)

if 'trk_charge' in data.columns:
   for feat in ['ktf_ecal_cluster_dphi', 'ktf_hcal_cluster_dphi', 'preid_trk_ecal_Dphi']:
      if feat in data.columns:
         data[feat] = data[feat]*data['trk_charge']

for c in data.columns:
   if data[c].dtype == np.dtype('bool') and c not in labelling:
      data[c] = data[c].astype(int) # convert bools to integers

################################################################################
print "### Load binning and save weights ..."

# File path
kmeans_model = '%s/kmeans_%s_weighter.pkl'%(mods, args.dataset)
if not os.path.isfile(kmeans_model):
   raise ValueError('I could not find the appropriate model, I have %s/kmeans_%s_weighter.pkl' % (mods, args.dataset))

# Cluster (bin) number vs trk (pt,eta)
kmeans = joblib.load(kmeans_model)
cluster = kmeans.predict(data[['log_trkpt','trk_eta']])

# Weights per cluster (bin) number
str_weights = json.load(open(kmeans_model.replace('.pkl','.json')))
weights = {}
for i in str_weights:
   try:
      weights[int(i)] = str_weights[i]
   except:
      pass

# Apply weight according to cluster (bin) number
apply_weight = np.vectorize(lambda x, y: y.get(x), excluded={2})
weights = apply_weight(cluster, weights)

# Apply weights to DF
data['weight'] = weights*np.invert(data.is_e) + data.is_e

################################################################################
print "### Split into low-pT and EGamma data frames ..."

if args.verbose :
   print data.describe()
   print data.info()

egamma = data[data.is_egamma]          # EGamma electrons
orig = data.copy()                     # all electrons
data = data[np.invert(data.is_egamma)] # low pT electrons
if args.verbose :
   print "orig.shape",orig.shape
   print "lowpt.shape",data.shape
   print "egamma.shape",egamma.shape

################################################################################
print "### Define train/test/validation data sets ..."

def train_test_split(data, div, thr):
   mask = data.evt % div
   mask = mask < thr
   return data[mask], data[np.invert(mask)]

train_test, validation = train_test_split(data, 10, 8)
train, test = train_test_split(train_test, 10, 6)

if args.verbose :
   debug(orig,'original')
   debug(train,'train')
   debug(test,'test')
   debug(validation,'validation')
   debug(egamma,'egamma',is_egamma=True)

################################################################################

clf = None
early_stop_kwargs = None

if args.train :
   print "### Train model ..."
   if args.config :
      print "Using parameters from config arg..."
      cfg = json.load(open(args.config))
      clf = xgb.XGBClassifier(
         # general parameters
         booster = 'gbtree',
         silent = False,
         nthread = args.nthreads,
         # learning task parameters
         objective = 'binary:logitraw',
         # booster parameters
         n_estimators = cfg['n_estimators'],
         learning_rate = cfg['learning_rate'],
         min_child_weight = cfg['min_child_weight'],
         max_depth = cfg['max_depth'],
         gamma = cfg['gamma'],
         subsample = cfg['subsample'],
         colsample_bytree = cfg['colsample_bytree'],
         reg_lambda = cfg['reg_lambda'],
         reg_alpha = cfg['reg_alpha'],
      )
   else :
      print "Using default parameters..."
      clf = xgb.XGBClassifier(
         # general parameters
         booster = 'gbtree',
         silent = False,
         nthread = args.nthreads,
         # learning task parameters
         objective = 'binary:logitraw',
         # booster parameters: use defaults
      )

   early_stop_kwargs = {} if args.no_early_stop else {
      'eval_set' : [(train[features].values, train.is_e.values.astype(int)),
                    (test[features].values, test.is_e.values.astype(int)),],
      'eval_metric' : ['error','auc'],
      'early_stopping_rounds' : 10
   }
   
   clf.fit(
      train[features].values, 
      train.is_e.values.astype(int), 
      sample_weight=train.weight.values,
      **early_stop_kwargs
   )

   full_model = '%s/%s_%s_%s_BDT.pkl' % (mods_, args.dataset, args.jobtag, args.what)
   joblib.dump(clf, full_model, compress=True)

   print '### Training done ...'

################################################################################
print "### Loading model ..."

full_model = '%s/%s_%s_%s_BDT.pkl' % (mods_, args.dataset, args.jobtag, args.what)
clf = joblib.load(full_model)
training_out = clf.predict_proba(validation[features].values)[:, 1]
with pd.option_context('mode.chained_assignment', None) : # Suppresses SettingwithCopyWarning
   validation['training_out'] = training_out

early = {}
if early_stop_kwargs is not None :
   early = early_stop_kwargs.copy()
   del early['eval_set']
args_dict = dict( clf.get_params().items() + early.items() + vars(args).items() )

with open('%s/%s_%s_%s_BDT.json' % (mods_, args.dataset, args.jobtag, args.what), 'w') as info :
   json.dump(args_dict, info)
   print 'args_dict',args_dict

if args.xml :
   print '### Converting to XML ...'
   xml = full_model.replace('.pkl', '.xml')
   clf = joblib.load(full_model)
   xgb_feats = ['f%d' % i for i in range(len(features))]
   convert_model(clf._Booster.get_dump(), zip(xgb_feats, cycle('F')), xml)
   xml_str = open(xml).read()
   for idx, feat in reversed(list(enumerate(features))):
      xml_str = xml_str.replace('f%d' % idx, feat)
   with open(xml.replace('.xml', '.fixed.xml'), 'w') as XML:
      XML.write(xml_str)

################################################################################
print "### Plotting ..."

plotting(plots,args.dataset,args,validation,egamma,data)
