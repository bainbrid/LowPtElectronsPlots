#pip install -U pandas-profiling
#import pandas_profiling
#df.profile_report()

################################################################################
# Imports ...

# python 2 and 3 compatibility
# pip install future and six
from __future__ import print_function
#from __future__ import absolute_import
import builtins
import future
from future.utils import raise_with_traceback
import past
import six

from argparse import ArgumentParser
import json
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib
import socket
#from tabulate import tabulate
import xgboost as xgb
import uproot

import matplotlib
matplotlib.use('Agg') # choose backend before doing anything else with pyplot! ('macosx')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties

#for x in range(0,20) : print(x,poisson_interval(x))

################################################################################
print("##### Command line args #####")

parser = ArgumentParser()
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--debug',action='store_true')
parser.add_argument('--nevents',default=-1,type=int)
parser.add_argument('--train',action='store_true')
parser.add_argument('--config',default='hyperparameters.json',type=str)
parser.add_argument('--nthreads', default=8, type=int)
args = parser.parse_args()
print("Command line args:",vars(args))

################################################################################
print("##### Define inputs #####")

print(os.getcwd())
assert os.getcwd().endswith("LowPtElectronsPlots/scripts"), print("You must execute this script from within the 'LowPtElectronsPlots/scripts' dir!")

print("hostname:",socket.gethostname())
lxplus = True if "cern.ch" in socket.gethostname() else False

# I/O directories
if lxplus :
   #input_data='/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD'
   input_data='~/eos/electrons/ntuples/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/2020May20'
else :
   input_data='../data'
print("input_data:",input_data)
if lxplus :
   cmssw_base='/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15'
   input_base='{:s}/src/LowPtElectrons/LowPtElectrons/macros/input'.format(cmssw_base)
   output_base='{:s}/src/LowPtElectrons/LowPtElectrons/macros/output'.format(cmssw_base)
else :
   input_base=os.getcwd()+"/../input"
   output_base=os.getcwd()+"/../output"
if not os.path.isdir(input_base) : 
   raise_with_traceback(ValueError('Could not find input_base "{:s}"'.format(input_base)))
print("input_base:",input_base)
if not os.path.isdir(output_base) : 
   os.makedirs(output_base)
print("output_base:",output_base)
   
files = [
   #input_data+'/output.LATEST.root', # 1,797,425 entries
   #input_data+'/2020May20/output_'+['aod.root','aod_test.root','miniaod.root','miniaod_test.root'][3]
   #input_data+'/temp_'+['miniaod.root','miniaod_test.root'][1]
   #input_data+'/output_numEvent1000.root'
   #input_data+'/test_nonres_med/output_0.root',
   input_data+'/output_0.root',
   #input_data+'/output_1.root',
   #input_data+'/output_2.root',
   #input_data+'/output_3.root',
   #input_data+'/test_res_small/output_1.root',
   #input_data+'/test_res_med/output_1.root',
   #input_data+'/test_nonres_med/output_1.root'
   ]
new_ntuples = any([ "LATEST" in x for x in files ])

features = [ # ORDER IS VERY IMPORTANT ! 
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
   'gsf_bdtout1'
]

additional = [
   'gen_pt','gen_eta', 
   'trk_pt','trk_eta','trk_charge','trk_dr',
   'gsf_pt','gsf_eta','gsf_dr','gsf_bdtout2','gsf_mode_pt',
   'ele_pt','ele_eta','ele_dr',
   'ele_mva_value','ele_mva_value_old','ele_mva_value_retrained',
   'ele_mva_value_depth10','ele_mva_value_depth11','ele_mva_value_depth13','ele_mva_value_depth15',
   'evt','weight','rho',
   'tag_pt','tag_eta',
   'gsf_dxy','gsf_dz','gsf_nhits','gsf_chi2red',
]

labelling = [
   'is_e','is_egamma',
   'has_trk','has_seed','has_gsf','has_ele',
   'seed_trk_driven','seed_ecal_driven'
]

columns = features + additional + labelling
#if new_ntuples is True : columns += ['pfgsf_pt','pfgsf_eta','has_pfgsf',]
columns = list(set(columns))

################################################################################
print("##### Load files #####")

def get_data(files,columns,features) :

   #@@ BEGIN
   pfgsf = ['pfgsf_pt','pfgsf_eta','has_pfgsf','pfgsf_mode_pt']
   has_pfgsf_branches = None
   try : 
      uproot.open(files[0]).get('ntuplizer/tree').pandas.df(branches=pfgsf)
      columns += pfgsf
      has_pfgsf_branches = True
   except KeyError : 
      has_pfgsf_branches = False
      print("Cannot find following branches:",pfgsf)
   except :
      print("Unknown error in get_data()")
      quit()
   #@@ END

   print('Getting files:\n', '\n'.join(files))
   dfs = [ uproot.open(i).get('ntuplizer/tree').pandas.df(branches=columns)  for i in files ]
   print('Extracted branches: ',columns)
   df = pd.concat(dfs)

   #@@ BEGIN
   if has_pfgsf_branches is False :
      columns += pfgsf
      for br in pfgsf : df[br] = df[br.replace('pfgsf','gsf')]
   #@@ END

   print('Available branches: ',df.keys())
   print('Features for model: ',features)
   if args.nevents > 0 : 
      print("Considering only first {:.0f} events ...".format(args.nevents))
      df = df.head(args.nevents)
   return df,has_pfgsf_branches

data,has_pfgsf_branches = get_data(files,columns,features)
print(data.columns)
print(data.dtypes)

################################################################################
print("##### Preprocessing the data #####")

# swap normal and mode estimates of GSF pT
if False :
   gsf_tmp_pt = list(data.gsf_pt)
   data.gsf_pt = data.gsf_mode_pt
   data.gsf_mode_pt = gsf_tmp_pt
   pfgsf_tmp_pt = list(data.pfgsf_pt)
   data.pfgsf_pt = data.pfgsf_mode_pt
   data.pfgsf_mode_pt = pfgsf_tmp_pt

print("gsf_pt:      "," ".join(["{:6.3f}".format(x) for x in data.gsf_pt if x > -10.][:10]))
print("gsf_mode_pt: "," ".join(["{:6.3f}".format(x) for x in data.gsf_mode_pt if x > -10.][:10]))

# Filter based on tag muon pT and eta
tag_muon_pt = 7.
tag_muon_eta = 1.5
print("Tag-side muon req, pT threshold:   ",tag_muon_pt)
print("Tag-side muon req, eta threshold:  ",tag_muon_eta)
print("Pre  tag-side muon req, data.shape:",data.shape)
data = data[(data.tag_pt>tag_muon_pt)&(np.abs(data.tag_eta)<tag_muon_eta)]
print("Post tag-side muon req, data.shape:",data.shape)

################################################################################

if args.train :

   print("##### Load binning and save weights #####")

   # print("Preprocessing some columns ...")
   log_trkpt = np.log10(data.trk_pt)
   log_trkpt[np.isnan(log_trkpt)] = -9999
   data['log_trkpt'] = log_trkpt
   data['prescale'] = data.weight
   data['weight'] = np.ones(data.weight.shape)

   # File path
   kmeans_model = '{:s}/binning_kmeans_3.pkl'.format(input_base)
   if not os.path.isfile(kmeans_model) :
      raise_with_traceback(ValueError('Could not find the model file "{:s}"'.format(kmeans_model)))
      # raise ValueError('Could not find the model file "%s"'%kmeans_model)
                    
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
print("##### Split into low-pT and EGamma data frames #####")

egamma = data[data.is_egamma]           # EGamma electrons
lowpt = data[np.invert(data.is_egamma)] # low pT electrons

print("total.shape",data.shape)
print("egamma.shape",egamma.shape)
print("lowpt.shape",lowpt.shape)
if args.verbose :
   pd.options.display.max_columns=None
   pd.options.display.width=None
   print(lowpt.describe().T)
   print(lowpt.info())
   #pretty=lambda df:tabulate(df,headers='keys',tablefmt='psql') # 'html'
   #print(pretty(lowpt.describe().T)))

################################################################################
print("##### Define train/validation/test data sets #####")

def train_test_split(data, div, thr):
   mask = data.evt % div
   mask = mask < thr
   return data[mask], data[np.invert(mask)]
   
train, validation, test = None, None, None
if args.train : 
   temp, test = train_test_split(lowpt, 100, 8) # _, 20%
   train, validation = train_test_split(temp, 10, 6) # 60%, 20%
else :
   temp, test = train_test_split(lowpt, 100, 2) # _, 98%
   train, validation = train_test_split(temp, 100, 1) # 1%, 1%

print("train.shape",train.shape)
print("validation.shape",validation.shape)
print("test.shape",test.shape)

if args.train : 
   df = train
   mask = (df.trk_pt > 0.5) & (df.trk_pt < 15.) & (np.abs(df.trk_eta) < 2.4) & (df.gsf_pt > 0.) 
   train = train[mask]
#   if args.debug :
#      df = validation
#      mask = (df.trk_pt > 0.5) & (df.trk_pt < 15.) & (np.abs(df.trk_eta) < 2.4) & (df.gsf_pt > 0.) 
#      validation = validation[mask]
#      df = test
#      mask = (df.trk_pt > 0.5) & (df.trk_pt < 15.) & (np.abs(df.trk_eta) < 2.4) & (df.gsf_pt > 0.) 
#      test = test[mask]

def debug(df,str=None,is_egamma=False) :
   if str is not None : print(str)
   elif is_egamma : print("EGAMMA")
   else : print("LOW PT")
   has_trk = (df.has_trk) & (df.trk_pt>0.5) & (np.abs(df.trk_eta)<2.4)
   if is_egamma and new_ntuples is True : 
      has_gsf = (df.has_pfgsf) & (df.pfgsf_pt>0.5) & (np.abs(df.pfgsf_eta)<2.4)
   else :
      has_gsf = (df.has_gsf) & (df.gsf_pt>0.5) & (np.abs(df.gsf_eta)<2.4)
   has_ele = (df.has_ele) & (df.ele_pt>0.5) & (np.abs(df.ele_eta)<2.4)
   print(pd.crosstab(df.is_e,
                     [has_trk,has_gsf,has_ele],
                     rownames=['is_e'],
                     colnames=['has_trk','has_pfgsf' if is_egamma and new_ntuples else 'has_gsf','has_ele'],
                     margins=True))
   print

if args.verbose :
   debug(data,'original')
   debug(train,'train')
   debug(validation,'validation')
   debug(test,'test')
   debug(egamma,'egamma',is_egamma=True)

################################################################################
print("##### Training #####")

model = None
results = None
early_stop_kwargs = None

if args.train :

   print("##### Define model #####")
   params = "{:s}/{:s}".format(input_base,args.config)
   if not os.path.isfile(params) :
      raise_with_traceback(ValueError('Could not find the hyperparameters file "{:s}"'.format(params)))
      #raise ValueError('Could not find the hyperparameters file "%s"'%params)
   cfg = json.load(open(params))
   model = xgb.XGBClassifier(
      objective = 'binary:logitraw',
      silent = False,
      verbose_eval=True,
      nthread = args.nthreads,
      booster = 'gbtree',
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

   print("##### Train model #####")
   #print(train[features].columns())
   model.fit(
      train[features].values, 
      train.is_e.values.astype(int), 
      sample_weight=train.weight.values,
      eval_set=[(train[features].values, train.is_e.values.astype(int)),
                (validation[features].values, validation.is_e.values.astype(int)),
                (test[features].values, test.is_e.values.astype(int))],
      eval_metric=['logloss','error','auc'],
      early_stopping_rounds=10
   )
   results = model.evals_result()

   model_file = '{:s}/model.pkl'.format(output_base)
   joblib.dump(model, model_file, compress=True)
   print("Write model to: ",model_file)
   print('### Training complete ...')

else : ## --train not specified

   print("##### Read trained model #####")
   model_file = '{:s}/model.pkl'.format(input_base)
   model = joblib.load(model_file)
   print("Read model from: ",model_file)
   model_file = '{:s}/model.pkl'.format(output_base)
   joblib.dump(model, model_file, compress=True)
   print("Write model to: ",model_file)
   print('### Loaded pre-existing model ...')

################################################################################
print("##### Added predictions to test set #####")

#training_out = model.predict_proba(test[features].as_matrix())[:,1]
#test['training_out'] = training_out

if args.debug :
   training_out = model.predict_proba(test[features].as_matrix())[:,1]
   test['training_out'] = training_out
   training_out = model.predict_proba(train[features].as_matrix())[:,1]
   train['training_out'] = training_out
   training_out = model.predict_proba(validation[features].as_matrix())[:,1]
   validation['training_out'] = training_out

################################################################################
print("##### Plotting ... #####")

#from plotting.scikitplots import scikitplots
#scikitplots(test,egamma)

#@@ override! should be commented normally ...
#has_pfgsf_branches=False
print("has_pfgsf_branches",has_pfgsf_branches)

# plot AxE or Eff?
AxE = True

# Reproduces Mauro's plot of ROC curves 

#from plotting.mauro import mauro
#mauro("../output/plots_train2/mauro",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

# My Original plotting scripts 

#from plotting.seed import seed
#seed("../output/plots_train2/seed",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

#from plotting.id import id
#id("../output/plots_train2/id",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

#from plotting.miniaod import miniaod
#miniaod("../output/plots_train2/miniaod",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

# Performance for BParking

#from plotting.bparking_trk import bparking_trk
#bparking_trk("../output/plots_train2/bparking_trk",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

#from plotting.bparking_gsf import bparking_gsf
#bparking_gsf("../output/plots_train2/bparking_gsf",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

#from plotting.bparking_binned import bparking_binned
#bparking_binned("../output/plots_train2/bparking_binned",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

#from plotting.bparking_trigger import bparking_trigger
#bparking_trigger("../output/plots_train2/bparking_trigger",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

# Performance for UltraLegacy

#from plotting.ultralegacy_vloose import ultralegacy_vloose
#ultralegacy_vloose("../output/plots_train2/ultralegacy_vloose",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

#from plotting.ultralegacy_tight import ultralegacy_tight
#ultralegacy_tight("../output/plots_train2/ultralegacy_tight",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

# BParking performance plot for CMS week (Otto's talk)

#from plotting.cmsweek import cmsweek
#cmsweek("../output/plots_train2/cmsweek",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

# Miscellaneous

#from plotting.bparking_dev import bparking_dev
#bparking_dev("../output/plots_train2/bparking_dev",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

#from plotting.bparking_dev2 import bparking_dev2
#bparking_dev2("../output/plots_train2/bparking_dev2",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

#from plotting.bparking_dev3 import bparking_dev3
#bparking_dev3("../output/plots_train2/bparking_dev3",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

#from plotting.bparking_dev4 import bparking_dev4
#bparking_dev4("../output/plots_train2/bparking_dev4",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)

from plotting.bparking_dev5 import bparking_dev5
bparking_dev5("../output/plots_train2/bparking_dev5",test,egamma,has_pfgsf_branches=has_pfgsf_branches,AxE=AxE)
