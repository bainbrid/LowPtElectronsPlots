#pip install -U pandas-profiling
#import pandas_profiling
#df.profile_report()

################################################################################
# Imports ...

# python 2 and 3 compatibility
# pip install future and six
from __future__ import print_function
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
from sklearn.metrics import roc_curve, roc_auc_score
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
assert os.getcwd().endswith("icenet/standalone/scripts"), print("You must execute this script from within the 'icenet/standalone/scripts' dir!")

print("hostname:",socket.gethostname())
lxplus = True if "cern.ch" in socket.gethostname() else False

# I/O directories
if lxplus :
   input_data='/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD'
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
   input_data+'/output.LATEST.root', # 1,797,425 entries
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
   'gsf_pt','gsf_eta','gsf_dr','gsf_bdtout2',
   'ele_pt','ele_eta','ele_dr','ele_mva_value','ele_mva_id',
   'evt','weight'
]

labelling = [
   'is_e','is_other','is_egamma',
   'has_trk','has_seed','has_gsf','has_ele',
   'seed_trk_driven','seed_ecal_driven'
]

columns = features + additional + labelling
#if new_ntuples is True : 
columns += ['pfgsf_pt','pfgsf_eta','has_pfgsf',]
columns = list(set(columns))

################################################################################
print("##### Load files #####")

def get_data(files,columns,features) :
   print('Getting files:\n', '\n'.join(files))
   dfs = [ uproot.open(i).get('ntuplizer/tree').pandas.df(branches=columns)  for i in files ]
   print('Extracted branches: ',columns)
   df = pd.concat(dfs)
   print('Available branches: ',df.keys())
   print('Features for model: ',features)
   if args.nevents > 0 : 
      print("Considering only first {:.0f} events ...".format(args.nevents))
      df = df.head(args.nevents)
   return df

data = get_data(files,columns,features)
print(data.columns)
print(data.dtypes)
print(data.shape)

print("Preprocessing some columns ...")
log_trkpt = np.log10(data.trk_pt)
log_trkpt[np.isnan(log_trkpt)] = -9999
data['log_trkpt'] = log_trkpt
data['prescale'] = data.weight
data['weight'] = np.ones(data.weight.shape)

################################################################################
print("##### Load binning and save weights #####")

# File path
kmeans_model = '{:s}/binning_kmeans_3.pkl'.format(input_base)
if not os.path.isfile(kmeans_model) :
   raise_with_traceback(ValueError('Could not find the model file "{:s}"'.format(kmeans_model)))
   #raise ValueError('Could not find the model file "%s"'%kmeans_model)
                    
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

print("data.shape",data.shape)
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
   
temp, test = train_test_split(lowpt, 10, 8)
train, validation = train_test_split(temp, 10, 6)

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
   print('### Training complete ...')

else : ## --train not specified

   print("##### Reading trained model #####")
   model_file = '{:s}/model.pkl'.format(input_base)
   model = joblib.load(model_file)
   model_file = '{:s}/model.pkl'.format(output_base)
   joblib.dump(model, model_file, compress=True)
   print('### Loaded pre-existing model ...')

################################################################################
print("##### Added predictions to test set #####")

training_out = model.predict_proba(test[features].as_matrix())[:,1]
test['training_out'] = training_out

if args.debug :
   training_out = model.predict_proba(train[features].as_matrix())[:,1]
   train['training_out'] = training_out
   training_out = model.predict_proba(validation[features].as_matrix())[:,1]
   validation['training_out'] = training_out

################################################################################
################################################################################
################################################################################
# utility method to add ROC curve to plot 

def plot( string, plt, draw_roc, draw_eff, 
          df, selection, discriminator, mask, 
          label, color, markersize, linestyle, linewidth=1.0 ) :
   
   if draw_roc is True and discriminator is None : 
      print("No discriminator given for ROC curve!")
      quit()

   if mask is None : mask = [True]*df.shape[0]
   denom = df.is_e#[mask]; 
   numer = denom & selection#[mask]
   eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   print("   eff/numer/denom: {:6.4f}".format(eff), numer.sum(), denom.sum())
   denom = ~df.is_e[mask]; numer = denom & selection[mask]
   mistag = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   print("    fr/numer/denom: {:6.4f}".format(mistag), numer.sum(), denom.sum())

   if draw_roc :
      roc = roc_curve(df.is_e[selection&mask], discriminator[selection&mask])
      auc = roc_auc_score(df.is_e[selection&mask], discriminator[selection&mask])
      plt.plot(roc[0]*mistag,
               roc[1]*eff,
               linestyle=linestyle,
               linewidth=linewidth,
               color=color, 
               label=label+', AUC: {:.3f}'.format(auc))
      plt.plot([mistag], [eff], marker='o', color=color, markersize=markersize)
      return eff,mistag,roc
   elif draw_eff :
      plt.plot([mistag], [eff], marker='o', color=color, markersize=markersize, 
               label=label)
      return eff,mistag,None

################################################################################
print("##### Plotting ROC curves #####")

plt.figure() #plt.figure(figsize=[8, 12])
ax = plt.subplot(111)
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width, box.height*0.666])
plt.title('')
plt.plot(np.arange(0.,1.,0.01),np.arange(0.,1.,0.01),'k--')
ax.tick_params(axis='x', pad=10.)
if lxplus :
   ax.text(0, 1, '\\textbf{CMS} \\textit{Simulation} \\textit{Preliminary}', 
           ha='left', va='bottom', transform=ax.transAxes)
else :
   ax.text(0, 1, 'CMS Simulation Preliminary', 
           ha='left', va='bottom', transform=ax.transAxes)
ax.text(1, 1, r'13 TeV', 
        ha='right', va='bottom', transform=ax.transAxes)
#plt.tight_layout()

if new_ntuples is True : 
   has_gsf = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.4)
else : 
   has_gsf = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.4)
has_ele = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.4)

# EGamma PF GSF track
print() 
eff1,fr1,_ = plot( string="EGamma GSF trk, AxE",
                   plt=plt, draw_roc=False, draw_eff=True,
                   df=egamma, selection=has_gsf, discriminator=None, mask=None,
                   label='EGamma GSF track ($\mathcal{A}\epsilon$)',
                   color='green', markersize=8, linestyle='solid',
)

# EGamma PF ele 
print() 
plot( string="EGamma PF ele, AxE",
      plt=plt, draw_roc=False, draw_eff=True,
      df=egamma, selection=has_ele, discriminator=None, mask=None,
      label='EGamma PF ele ($\mathcal{A}\epsilon$)',
      color='purple', markersize=8, linestyle='solid',
)

has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.4)
has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.4)

# Low-pT GSF electrons (PreId unbiased)
print() 
plot( string="Low pT GSF trk (PreId), AxE",
      plt=plt, draw_roc=True, draw_eff=False,
      df=test, selection=has_gsf, discriminator=test.gsf_bdtout1, mask=None,
      label='Low-$p_{T}$ GSF track + unbiased ($\mathcal{A}\epsilon$)',
      color='red', markersize=8, linestyle='dashed',
)

# Low-pT GSF electrons (CMSSW)
print() 
plot( string="Low pT ele (CMSSW), AxE",
      plt=plt, draw_roc=True, draw_eff=False,
      df=test, selection=has_ele, discriminator=test.ele_mva_value, mask=None,
      label='Low-$p_{T}$ ele + 2019Jun28 model ($\mathcal{A}\epsilon$)',
      color='blue', markersize=8, linestyle='dashdot',
)

# Low-pT GSF electrons (retraining)
print() 
eff2,fr2,roc2 = plot( string="Low pT ele (latest), AxE",
                      plt=plt, draw_roc=True, draw_eff=False,
                      df=test, selection=has_ele, discriminator=test.training_out, mask=None,
                      label='Low-$p_{T}$ ele + latest model ($\mathcal{A}\epsilon$)',
                      color='blue', markersize=8, linestyle='solid',
)

roc = (roc2[0]*fr2,roc2[1]*eff2,roc2[2]) 
idxL = np.abs(roc[0]-fr1).argmin()
idxT = np.abs(roc[1]-eff1).argmin()
print("   PFele: eff/fr/thresh:","{:.3f}/{:.4f}/{:4.2f} ".format(eff1,fr1,np.nan))
print("   Loose: eff/fr/thresh:","{:.3f}/{:.4f}/{:4.2f} ".format(roc[1][idxL],roc[0][idxL],roc[2][idxL]))
print("   Tight: eff/fr/thresh:","{:.3f}/{:.4f}/{:4.2f} ".format(roc[1][idxT],roc[0][idxT],roc[2][idxT]))

# Adapt legend
def update_prop(handle, orig):
   handle.update_from(orig)
   handle.set_marker("o")
plt.legend(handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)})

plt.xlabel('Mistag rate')
if lxplus : plt.ylabel(r'Acceptance $\times$ efficiency')
else : plt.ylabel(r'Acceptance x efficiency')
plt.legend(loc='best')
#plt.legend(loc='lower left', bbox_to_anchor=(0., 1.1)) 
plt.ylim(0., 1)
plt.gca().set_xscale('log')
plt.xlim(1e-4, 1)

#try : plt.savefig('{:s}/roc.png'.format(output_base))
#except : print('Issue: {:s}/roc.png'.format(output_base))
try : plt.savefig('{:s}/roc.pdf'.format(output_base))
except : print('Issue: {:s}/roc.pdf'.format(output_base))
plt.clf()

#################################################################################
print("##### Plotting loss and AUC curves #####")

if results is not None :
   print(results)
   epochs = len(results['validation_0']['auc'])
   x_axis = range(0, epochs)

   # AUC 
   fig, ax = plt.subplots()
   ax.plot(x_axis, results['validation_0']['auc'], label='Train')
   ax.plot(x_axis, results['validation_1']['auc'], label='Valid')
   ax.plot(x_axis, results['validation_2']['auc'], label='Test')
   ax.legend()
   plt.xlabel('epoch')
   plt.ylabel('AUC')
   plt.title('XGBClassifier')
   plt.grid(True)
   plt.ylim(0.,1.)
   #plt.tight_layout()
   #try : plt.savefig('{:s}/auc.png'.format(output_base))
   #except : print('Issue: {:s}/auc.png'.format(output_base))
   try : plt.savefig('{:s}/auc.pdf'.format(output_base))
   except : print('Issue: {:s}/auc.pdf'.format(output_base))

   # Classification error 
   fig, ax = plt.subplots()
   ax.plot(x_axis, results['validation_0']['error'], label='Train')
   ax.plot(x_axis, results['validation_1']['error'], label='Valid')
   ax.plot(x_axis, results['validation_2']['error'], label='Test')
   ax.legend()
   plt.xlabel('epoch')
   plt.ylabel('Classification error')
   plt.title('XGBClassifier')
   plt.grid(True)
   plt.ylim(bottom=0.)
   #plt.tight_layout()
   #try : plt.savefig('{:s}/error.png'.format(output_base))
   #except : print('Issue: {:s}/error.png'.format(output_base))
   try : plt.savefig('{:s}/error.pdf'.format(output_base))
   except : print('Issue: {:s}/error.pdf'.format(output_base))

   # Log loss 
   fig, ax = plt.subplots()
   ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
   ax.plot(x_axis, results['validation_1']['logloss'], label='Valid')
   ax.plot(x_axis, results['validation_2']['logloss'], label='Test', linestyle='dashed')
   ax.legend()
   plt.xlabel('epoch')
   plt.ylabel('Log loss')
   plt.title('XGBClassifier')
   plt.grid(True)
   plt.ylim(bottom=0.)
   #plt.tight_layout()
   #try : plt.savefig('{:s}/loss.png'.format(output_base))
   #except : print('Issue: {:s}/loss.png'.format(output_base))
   try : plt.savefig('{:s}/loss.pdf'.format(output_base))
   except : print('Issue: {:s}/loss.pdf'.format(output_base))
   plt.clf()

################################################################################
if args.debug :
   print("##### Plotting ROC curves based on train, valid, test #####")

   plt.figure()
   ax = plt.subplot(111)
   plt.title('')
   plt.plot(np.arange(0.,1.,0.01),np.arange(0.,1.,0.01),'k--')
   ax.tick_params(axis='x', pad=10.)
   if lxplus :
      ax.text(0, 1, '\\textbf{CMS} \\textit{Simulation} \\textit{Preliminary}', 
              ha='left', va='bottom', transform=ax.transAxes)
   else :
      ax.text(0, 1, 'CMS Simulation Preliminary', 
              ha='left', va='bottom', transform=ax.transAxes)
   ax.text(1, 1, r'13 TeV', 
           ha='right', va='bottom', transform=ax.transAxes)
      

   has_ele = (train.has_ele) & (train.ele_pt>0.5) & (np.abs(train.ele_eta)<2.4)
   eff2,fr2,roc2 = plot( string="Train",
                         plt=plt, draw_roc=True, draw_eff=False,
                         df=train, selection=has_ele, discriminator=train.training_out, mask=None,
                         label='Train',
                         color='blue', markersize=8, linestyle='solid',
                         )

   has_ele = (validation.has_ele) & (validation.ele_pt>0.5) & (np.abs(validation.ele_eta)<2.4)
   eff2,fr2,roc2 = plot( string="Validation",
                         plt=plt, draw_roc=True, draw_eff=False,
                         df=validation, selection=has_ele, discriminator=validation.training_out, mask=None,
                         label='Validation',
                         color='orange', markersize=8, linestyle='solid',
                         )

   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.4)
   eff2,fr2,roc2 = plot( string="Test",
                         plt=plt, draw_roc=True, draw_eff=False,
                         df=test, selection=has_ele, discriminator=test.training_out, mask=None,
                         label='Test',
                         color='green', markersize=8, linestyle='dashed',
                         )

   # Adapt legend
   def update_prop(handle, orig):
      handle.update_from(orig)
      handle.set_marker("o")
   plt.legend(handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)})

   plt.xlabel('Mistag rate')
   if lxplus : plt.ylabel(r'Acceptance $\times$ efficiency')
   else : plt.ylabel(r'Acceptance x efficiency')
   plt.legend(loc='best')
   plt.ylim(0., 1)
   plt.gca().set_xscale('log')
   plt.xlim(1e-4, 1)

   # try : plt.savefig('{:s}/roc.png'.format(output_base))
   # except : print('Issue: {:s}/roc.png'.format(output_base))
   try : plt.savefig('{:s}/roc_tvt.pdf'.format(output_base))
   except : print('Issue: {:s}/roc_tvt.pdf'.format(output_base))
   plt.clf()

################################################################################

def summary(name,ds) :
   print("{:10s}:   {:6.0f}".format(name,len(ds.is_e)))
   print("                is_e  !is_e")
   first  = (ds.is_e).sum()
   second = np.invert(ds.is_e).sum()
   print(" is_e?:       {:6.0f} {:6.0f} {:5.3f}".format(first,second,first*1./second if second > 0 else 0.))
   first  = ((ds.is_e)&(ds.trk_pt>0.)).sum()
   second = (np.invert(ds.is_e)&(ds.trk_pt>0.)).sum()
   print("  has_trk:    {:6.0f} {:6.0f} {:5.3f}".format(first,second,first*1./second if second > 0 else 0.))
   first  = ((ds.is_e)&(ds.trk_pt>0.)&(ds.gsf_pt>0.)).sum()
   second = (np.invert(ds.is_e)&(ds.trk_pt>0.)&(ds.gsf_pt>0.)).sum()
   print("   has_gsf:   {:6.0f} {:6.0f} {:5.3f}".format(first,second,first*1./second if second > 0 else 0.))
   first  = ((ds.is_e)&(ds.trk_pt>0.)&(ds.gsf_pt>0.)&(ds.gsf_bdtout1>3.05)).sum()
   second = (np.invert(ds.is_e)&(ds.trk_pt>0.)&(ds.gsf_pt>0.)&(ds.gsf_bdtout1>3.05)).sum()
   print("   Tight seed:{:6.0f} {:6.0f} {:5.3f}".format(first,second,first*1./second if second > 0 else 0.))
   first  = ((ds.is_e)&(ds.trk_pt>0.)&(ds.gsf_pt>0.)&(ds.gsf_bdtout1>3.05)&(ds.training_out>4.92)).sum()
   second = (np.invert(ds.is_e)&(ds.trk_pt>0.)&(ds.gsf_pt>0.)&(ds.gsf_bdtout1>3.05)&(ds.training_out>4.92)).sum()
   print("    Tight ID :{:6.0f} {:6.0f} {:5.3f}".format(first,second,first*1./second if second > 0 else 0.))

if args.debug :
   summary("Train",train)
   summary("Validation",validation)
summary("Test",test)

################################################################################

from plotting.discriminator import discriminator as discrim

if args.debug :
   discrim(output_base,
           title='Seeding, TRK pT > 0.5 GeV', 
           suffix='_seed',
           signal_train = train.gsf_bdtout1[(train.is_e)&(train.trk_pt>0.)],
           signal_test = test.gsf_bdtout1[(test.is_e)&(test.trk_pt>0.)],
           bkgd_train = train.gsf_bdtout1[np.invert(train.is_e)&(train.trk_pt>0.)],
           bkgd_test = test.gsf_bdtout1[np.invert(test.is_e)&(test.trk_pt>0.)],
           )
   discrim(output_base,
           title='ID, GSF pT > 0.5 GeV',
           signal_train = train.training_out[(train.is_e)&(train.gsf_pt>0.)],
           signal_test = test.training_out[(test.is_e)&(test.gsf_pt>0.)],
           bkgd_train = train.training_out[np.invert(train.is_e)&(train.gsf_pt>0.)],
           bkgd_test = test.training_out[np.invert(test.is_e)&(test.gsf_pt>0.)],
           )
   discrim(output_base,
           title='ID, 0.5 < GSF pT < 2 GeV',
           suffix='_low',
           signal_train = train.training_out[(train.is_e)&(train.gsf_pt>0.)&(train.gsf_pt<2.)],
           signal_test = test.training_out[(test.is_e)&(test.gsf_pt>0.)&(test.gsf_pt<2.)],
           bkgd_train = train.training_out[np.invert(train.is_e)&(train.gsf_pt>0.)&(train.gsf_pt<2.)],
           bkgd_test = test.training_out[np.invert(test.is_e)&(test.gsf_pt>0.)&(test.gsf_pt<2.)],
           )
   discrim(output_base,
           title='ID, GSF pT > 2 GeV',
           suffix='_high',
           signal_train = train.training_out[(train.is_e)&(train.gsf_pt>2.)],
           signal_test = test.training_out[(test.is_e)&(test.gsf_pt>2.)],
           bkgd_train = train.training_out[np.invert(train.is_e)&(train.gsf_pt>2.)],
           bkgd_test = test.training_out[np.invert(test.is_e)&(test.gsf_pt>2.)],
           )

################################################################################

# Seed BDT thresholds
# https://github.com/cms-sw/cmssw/blob/CMSSW_10_2_X/RecoEgamma/EgammaElectronProducers/python/lowPtGsfElectronSeeds_cfi.py#L4

# Seed efficiency for GSF electrons w.r.t. GEN electrons vs GEN pT 
my_effs1 = {
   "path":output_base,"suffix":"_seed","mistag":False,"title":"Seed efficiency vs GEN pT","histograms":False,
   "curves":{
      "vl_seed":{"legend":"Very Loose seeding","color":"orange", # Very Loose seeding (>0.19 | >-1.99)
                 "var":test.gen_pt,"mask":(test.is_e)&(test.trk_pt>0.5),"condition":(test.ele_pt>0.)},
      "tight_seed":{"legend":"Tight seeding","color":"green", # Tight seeding (3.05)
                    "var":test.gen_pt,"mask":(test.is_e)&(test.trk_pt>1.0),"condition":(test.ele_pt>0.)&(test.gsf_bdtout1>3.05)},
#      "loose_id":{"legend":"Loose ID","color":"blue", # Loose ID (4.24)
#                  "var":test.gen_pt,"mask":(test.is_e)&(test.trk_pt>0.5),"condition":(test.ele_pt>0.)&(test.training_out>4.24)},
      "pf_ele":{"legend":"PF electrons","color":"purple", # PF electrons
                "var":egamma.gen_pt,"mask":(egamma.is_e)&(egamma.trk_pt>0.5),"condition":(egamma.ele_pt>0.)},
      }
   }
      
# Seed fake rate for GSF electrons vs TRK pT 
my_fakes1 = {
   "path":output_base,"suffix":"_seed","mistag":True,"title":"Seed mistag rate vs TRK pT","histograms":False,
   "curves":{
      "vl_seed":{"legend":"Very Loose seeding","color":"orange", # Very Loose seeding (>0.19 | >-1.99)
                 "var":test.trk_pt,"mask":(~test.is_e)&(test.trk_pt>0.5),"condition":(test.ele_pt>0.)},
      "tight_seed":{"legend":"Tight seeding","color":"green", # Tight seeding (3.05)
                    "var":test.trk_pt,"mask":(~test.is_e)&(test.trk_pt>1.0),"condition":(test.ele_pt>0.)&(test.gsf_bdtout1>3.05)},
#      "loose_id":{"legend":"Loose ID","color":"blue", # Loose ID (4.24)
#                  "var":test.trk_pt,"mask":(~test.is_e)&(test.trk_pt>0.5),"condition":(test.ele_pt>0.)&(test.training_out>4.24)},
      "pf_ele":{"legend":"PF electrons","color":"purple", # PF electrons
                "var":egamma.trk_pt,"mask":(~egamma.is_e)&(egamma.trk_pt>0.5),"condition":(egamma.ele_pt>0.)},
      },
   }

# ID efficiency for GSF electrons w.r.t. GSF tracks vs GEN pT 
my_effs2 = {
   "path":output_base,"suffix":"_id","mistag":False,"title":"ID efficiency vs GEN pT","histograms":False,
   "curves":{
      "vl_seed":{"legend":"Very Loose seeding","color":"orange", # Very Loose seeding (>0.19 | >-1.99)
                 "var":test.gen_pt,"mask":(test.is_e)&(test.gsf_pt>0.5),"condition":(test.ele_pt>0.)},
      "tight_seed":{"legend":"Tight seeding","color":"green", # Tight seeding (3.05)
                    "var":test.gen_pt,"mask":(test.is_e)&(test.gsf_pt>1.0),"condition":(test.ele_pt>0.)&(test.gsf_bdtout1>3.05)},
      "loose_id":{"legend":"Loose ID","color":"blue", # Loose ID (4.24)
                  "var":test.gen_pt,"mask":(test.is_e)&(test.gsf_pt>0.5),"condition":(test.ele_pt>0.)&(test.training_out>4.24)},
      "pf_ele":{"legend":"PF electrons","color":"purple", # PF electrons
                "var":egamma.gen_pt,"mask":(egamma.is_e)&(egamma.pfgsf_pt>0.5),"condition":(egamma.ele_pt>0.)},
      }
   }
      
# ID fake rate for GSF electrons vs TRK pT 
my_fakes2 = {
   "path":output_base,"suffix":"_id","mistag":True,"title":"ID mistag rate vs TRK pT","histograms":False,
   "curves":{
      "vl_seed":{"legend":"Very Loose seeding","color":"orange", # Very Loose seeding (>0.19 | >-1.99)
                 "var":test.trk_pt,"mask":(~test.is_e)&(test.gsf_pt>0.5),"condition":(test.ele_pt>0.)},
      "tight_seed":{"legend":"Tight seeding","color":"green", # Tight seeding (3.05)
                    "var":test.trk_pt,"mask":(~test.is_e)&(test.gsf_pt>1.0),"condition":(test.ele_pt>0.)&(test.gsf_bdtout1>3.05)},
      "loose_id":{"legend":"Loose ID","color":"blue", # Loose ID (4.24)
                  "var":test.trk_pt,"mask":(~test.is_e)&(test.gsf_pt>0.5),"condition":(test.ele_pt>0.)&(test.training_out>4.24)},
      "pf_ele":{"legend":"PF electrons","color":"purple", # PF electrons
                "var":egamma.trk_pt,"mask":(~egamma.is_e)&(egamma.pfgsf_pt>0.5),"condition":(egamma.ele_pt>0.)},
      },
   }

from plotting.efficiencies import efficiencies as effs

my_effs = my_effs1
effs(path=my_effs["path"],suffix=my_effs["suffix"],mistag=my_effs["mistag"],
     title=my_effs["title"],histograms=my_effs["histograms"],curves=my_effs["curves"])
my_effs = my_fakes1
effs(path=my_effs["path"],suffix=my_effs["suffix"],mistag=my_effs["mistag"],
     title=my_effs["title"],histograms=my_effs["histograms"],curves=my_effs["curves"])

my_effs = my_effs2
effs(path=my_effs["path"],suffix=my_effs["suffix"],mistag=my_effs["mistag"],
     title=my_effs["title"],histograms=my_effs["histograms"],curves=my_effs["curves"])
my_effs = my_fakes2
effs(path=my_effs["path"],suffix=my_effs["suffix"],mistag=my_effs["mistag"],
     title=my_effs["title"],histograms=my_effs["histograms"],curves=my_effs["curves"])

################################################################################

my_effs = {

   # All electrons
   "all_electrons":{ 
      "mistag":False,
      "title":"AxE vs GEN pT, Very Loose seeding (>0.19 | >-1.99)",
      "suffix":"",
      "total" :test.gen_pt[(test.is_e)],
      "passed":test.gen_pt[(test.is_e)&(test.gsf_pt>0.)],
      },
   # All electrons, mistag
   "all_electrons_mistag":{ 
      "mistag":True,
      "title":"Mistag rate vs TRK pT, Very Loose seeding (>0.19 | >-1.99)",
      "suffix":"",
      "total" :test.trk_pt[np.invert(test.is_e)],
      "passed":test.trk_pt[np.invert(test.is_e)&(test.gsf_pt>0.)],
      },

   # Tight seeding threshold (3.05)
   # https://github.com/cms-sw/cmssw/blob/CMSSW_10_2_X/RecoEgamma/EgammaElectronProducers/python/lowPtGsfElectronSeeds_cfi.py#L4
   "tight_seed":{ 
      "mistag":False,
      "title":"AxE vs GEN pT, Tight seeding (>3.05)",
      "suffix":"_tight_seed",
      "total" :test.gen_pt[(test.is_e)],
      "passed":test.gen_pt[(test.is_e)&(test.gsf_pt>0.)&(test.gsf_bdtout1>3.05)],
      },
   # Tight seeding threshold (3.05), mistag
   # https://github.com/cms-sw/cmssw/blob/CMSSW_10_2_X/RecoEgamma/EgammaElectronProducers/python/lowPtGsfElectronSeeds_cfi.py#L4
   "tight_seed_mistag":{ 
      "mistag":True,
      "title":"Mistag rate vs TRK pT, Tight seeding (>3.05)",
      "suffix":"_tight_seed",
      "total" :test.trk_pt[np.invert(test.is_e)&(test.gsf_pt>0.)],
      "passed":test.trk_pt[np.invert(test.is_e)&(test.gsf_pt>0.)&(test.gsf_bdtout1>3.05)],
      },
   
   # Tight biased seeding threshold (2.42)
   # https://github.com/cms-sw/cmssw/blob/CMSSW_10_2_X/RecoEgamma/EgammaElectronProducers/python/lowPtGsfElectronSeeds_cfi.py#L4
   "tight_seed_or":{ 
      "mistag":False,
      "title":"AxE vs GEN pT, Tight biased seeding (>2.42)",
      "suffix":"_tight_seed_biased",
      "total" :test.gen_pt[(test.is_e)],
      "passed":test.gen_pt[(test.is_e)&(test.gsf_pt>0.)&(test.gsf_bdtout2>2.42)],
      },
   # Tight biased seeding threshold (2.42), mistag
   # https://github.com/cms-sw/cmssw/blob/CMSSW_10_2_X/RecoEgamma/EgammaElectronProducers/python/lowPtGsfElectronSeeds_cfi.py#L4
   "tight_seed_or_mistag":{ 
      "mistag":True,
      "title":"Mistag rate vs TRK pT, Tight biased seeding (>2.42)",
      "suffix":"_tight_seed_biased",
      "total" :test.trk_pt[np.invert(test.is_e)&(test.gsf_pt>0.)],
      "passed":test.trk_pt[np.invert(test.is_e)&(test.gsf_pt>0.)&(test.gsf_bdtout2>2.42)],
      },

   # 2019Aug07 Loose ID threshold (4.24)
   # https://www.dropbox.com/s/kd6w3wfcglhelpm/190807_ElectronID.pdf?dl=0
   "loose_id":{ 
      "mistag":False,
      "title":"AxE vs GEN pT, 2019Aug07 Loose ID (>4.24)",
      "suffix":"_loose_id",
      "total" :test.gen_pt[(test.is_e)],
      "passed":test.gen_pt[(test.is_e)&(test.gsf_pt>0.)&(test.training_out>4.24)],
      },
   # 2019Aug07 Loose ID threshold (4.24), mistag
   # https://www.dropbox.com/s/kd6w3wfcglhelpm/190807_ElectronID.pdf?dl=0
   "loose_id_mistag":{ 
      "mistag":True,
      "title":"Mistag rate vs GSF pT, 2019Aug07 Loose ID (>4.24)",
      "suffix":"_loose_id",
      "total" :test.gsf_pt[np.invert(test.is_e)&(test.gsf_pt>0.)],
      "passed":test.gsf_pt[np.invert(test.is_e)&(test.gsf_pt>0.)&(test.training_out>4.24)],
      },

   # 2019Aug07 Tight ID threshold (4.93)
   # https://www.dropbox.com/s/kd6w3wfcglhelpm/190807_ElectronID.pdf?dl=0
   "id_tight":{ 
      "mistag":False,
      "title":"AxE vs GEN pT, 2019Aug07 Tight ID (>4.93)",
      "suffix":"_tight_id",
      "total" :test.gen_pt[(test.is_e)],
      "passed":test.gen_pt[(test.is_e)&(test.gsf_pt>0.)&(test.training_out>4.93)],
      },
   # 2019Aug07 Tight ID threshold (4.93), mistag
   # https://www.dropbox.com/s/kd6w3wfcglhelpm/190807_ElectronID.pdf?dl=0
   "id_tight_mistag":{ 
      "mistag":True,
      "title":"Mistag rate vs GSF pT, 2019Aug07 Tight ID (>4.93)",
      "suffix":"_tight_id",
      "total" :test.gsf_pt[np.invert(test.is_e)&(test.gsf_pt>0.)],
      "passed":test.gsf_pt[np.invert(test.is_e)&(test.gsf_pt>0.)&(test.training_out>4.93)],
      },

   }

from plotting.efficiency import efficiency as eff
for key in my_effs.keys() :
   eff(output_base,
       mistag=my_effs[key]["mistag"],
       title=my_effs[key]["title"], 
       suffix=my_effs[key]["suffix"],
       value_total=my_effs[key]["total"],
       value_passed=my_effs[key]["passed"])
