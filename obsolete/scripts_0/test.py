################################################################################
# Imports ...

from argparse import ArgumentParser
import os
import uproot
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import json
#from tabulate import tabulate
import xgboost as xgb
import matplotlib
matplotlib.use('Agg') # choose backend before doing anything else with pyplot! 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties

################################################################################
print "##### Command line args #####"

parser = ArgumentParser()
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--nevents',default=-1,type=int)
parser.add_argument('--train',action='store_true')
parser.add_argument('--config',default='hyperparameters.json',type=str)
parser.add_argument('--nthreads', default=8, type=int)
args = parser.parse_args()
print "Command line args:",vars(args)

################################################################################
print "##### Define inputs #####"

cmssw_base='/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15'
base='%s/src/LowPtElectrons/LowPtElectrons/macros'%(cmssw_base)

files = [
#   '/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15/src/2-ntuples-from-crab/output.LATEST.root', # 1,797,425 entries
   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_0.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_1.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_2.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_3.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_4.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_5.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_6.root',
]
new_ntuples = any([ "LATEST" in x for x in files ])

output='%s/output/'%(base)
if not os.path.isdir(output) : os.makedirs(output)

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
   'is_e','is_egamma',
   'has_trk','has_seed','has_gsf','has_ele',
   'seed_trk_driven','seed_ecal_driven'
]

columns = features + additional + labelling
if new_ntuples is True : columns += ['pfgsf_pt','pfgsf_eta','has_pfgsf',]
columns = list(set(columns))

################################################################################
print "##### Load files #####"

def get_data(files,columns,features) :

   print 'Getting files:\n', '\n'.join(files)
   ntuples = [ uproot.open(i) for i in files ]
   print 'Available branches: ',ntuples[0]['ntuplizer/tree'].keys()
   print 'Extracted branches: ',columns
   print 'Features for model: ',features

   content = None
   for i,ntuple in enumerate(ntuples) :
      print 'Parsing contents of: ', files[i]
      try:
         arrays = ntuple['ntuplizer/tree'].arrays(columns)
      except KeyError as ex :
         print 'Exception! ', ex
         raise RuntimeError('Failed to open %s properly' % files[i])
      if i == 0 : 
         content = arrays
      else : 
         for column in columns : content[column] = np.concatenate((content[column],arrays[column]))
   #return content

#content = get_data(files,columns,features)

################################################################################
#print "##### Pre-process data #####"

   data = pd.DataFrame(content)
   if args.nevents > 0 : 
      print "Considering only first %s events ..."%args.nevents
      data = data.head(args.nevents)
   return data

data = get_data(files,columns,features)

print "Preprocessing some columns ..."
log_trkpt = np.log10(data.trk_pt)
log_trkpt[np.isnan(log_trkpt)] = -9999
data['log_trkpt'] = log_trkpt
data['prescale'] = data.weight
data['weight'] = np.ones(data.weight.shape)

print data.columns
print data.dtypes
print data.shape
quit()

################################################################################
print "##### Load binning and save weights #####"

# File path
kmeans_model = '%s/binning_kmeans.pkl'%(output)
if not os.path.isfile(kmeans_model) :
   raise ValueError('Could not find the model file "%s"'%kmeans_model)
                    
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
print "##### Split into low-pT and EGamma data frames #####"

egamma = data[data.is_egamma]           # EGamma electrons
lowpt = data[np.invert(data.is_egamma)] # low pT electrons

print "data.shape",data.shape
print "egamma.shape",egamma.shape
print "lowpt.shape",lowpt.shape
if args.verbose :
   pd.options.display.max_columns=None
   pd.options.display.width=None
   print lowpt.describe().T
   print lowpt.info()
   #pretty=lambda df:tabulate(df,headers='keys',tablefmt='psql') # 'html'
   #print(pretty(lowpt.describe().T))

################################################################################
print "##### Define train/validation/test data sets #####"

def train_test_split(data, div, thr):
   mask = data.evt % div
   mask = mask < thr
   return data[mask], data[np.invert(mask)]
   
temp, test = train_test_split(lowpt, 10, 8)
train, validation = train_test_split(temp, 10, 6)

def debug(df,str=None,is_egamma=False) :
   if str is not None : print str
   elif is_egamma : print "EGAMMA"
   else : print "LOW PT"
   has_trk = (df.has_trk) & (df.trk_pt>0.5) & (np.abs(df.trk_eta)<2.4)
   if is_egamma and new_ntuples is True : 
      has_gsf = (df.has_pfgsf) & (df.pfgsf_pt>0.5) & (np.abs(df.pfgsf_eta)<2.4)
   else :
      has_gsf = (df.has_gsf) & (df.gsf_pt>0.5) & (np.abs(df.gsf_eta)<2.4)
   has_ele = (df.has_ele) & (df.ele_pt>0.5) & (np.abs(df.ele_eta)<2.4)
   print pd.crosstab(df.is_e,
                     [has_trk,has_gsf,has_ele],
                     rownames=['is_e'],
                     colnames=['has_trk','has_pfgsf' if is_egamma and new_ntuples else 'has_gsf','has_ele'],
                     margins=True)
   print

if args.verbose :
   debug(data,'original')
   debug(train,'train')
   debug(validation,'validation')
   debug(test,'test')
   debug(egamma,'egamma',is_egamma=True)

################################################################################

model = None
early_stop_kwargs = None

if args.train :

   print "##### Define model #####"
   params = output+args.config
   if not os.path.isfile(params) :
      raise ValueError('Could not find the hyperparameters file "%s"'%params)
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

   print "##### Train model #####"
   #print train[features].columns()
   model.fit(
      train[features].values, 
      train.is_e.values.astype(int), 
      sample_weight=train.weight.values,
      eval_set=[(validation[features].values, validation.is_e.values.astype(int)),
                (validation[features].values, validation.is_e.values.astype(int))],
      eval_metric=['error','auc'],
      early_stopping_rounds=10
   )

   model_file = '%s/model.pkl'%output
   joblib.dump(model, model_file, compress=True)

   print '### Training complete ...'

else :

   print "##### Reading trained model #####"
   
################################################################################
print "##### Added predictions to test set #####"

training_out = model.predict_proba(test[features].as_matrix())[:,1]
test['training_out'] = training_out

################################################################################
# utility method to add ROC curve to plot 

def plot( string, plt, draw_roc, draw_eff, 
          df, selection, discriminator, mask, 
          label, color, markersize, linestyle, linewidth=1.0 ) :
   
   if draw_roc is True and discriminator is None : 
      print "No discriminator given for ROC curve!"
      quit()

   if mask is None : mask = [True]*df.shape[0]
   denom = df.is_e#[mask]; 
   numer = denom & selection#[mask]
   eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   print "   eff/numer/denom: {:6.4f}".format(eff), numer.sum(), denom.sum()
   denom = ~df.is_e[mask]; numer = denom & selection[mask]
   mistag = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   print "    fr/numer/denom: {:6.4f}".format(mistag), numer.sum(), denom.sum()

   if draw_roc :
      roc = roc_curve(df.is_e[selection&mask], discriminator[selection&mask])
      auc = roc_auc_score(df.is_e[selection&mask], discriminator[selection&mask])
      plt.plot(roc[0]*mistag,
               roc[1]*eff,
               linestyle=linestyle,
               linewidth=linewidth,
               color=color, 
               label=label+', AUC: %.3f'%auc)
      plt.plot([mistag], [eff], marker='o', color=color, markersize=markersize)
      return eff,mistag,roc
   elif draw_eff :
      plt.plot([mistag], [eff], marker='o', color=color, markersize=markersize, 
               label=label)
      return eff,mistag,None

################################################################################
print "##### Plotting #####"

plt.figure() #plt.figure(figsize=[8, 12])
ax = plt.subplot(111)
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width, box.height*0.666])
plt.title('')
plt.plot(np.arange(0.,1.,0.01),np.arange(0.,1.,0.01),'k--')
ax.tick_params(axis='x', pad=10.)
ax.text(0, 1, '\\textbf{CMS} \\textit{Simulation} \\textit{Preliminary}', 
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
print 
eff1,fr1,_ = plot( string="EGamma GSF trk, AxE",
                   plt=plt, draw_roc=False, draw_eff=True,
                   df=egamma, selection=has_gsf, discriminator=None, mask=None,
                   label='EGamma GSF track ($\mathcal{A}\epsilon$)',
                   color='green', markersize=8, linestyle='solid',
)

# EGamma PF ele 
print 
plot( string="EGamma PF ele, AxE",
      plt=plt, draw_roc=False, draw_eff=True,
      df=egamma, selection=has_ele, discriminator=None, mask=None,
      label='EGamma PF ele ($\mathcal{A}\epsilon$)',
      color='purple', markersize=8, linestyle='solid',
)

has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.4)
has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.4)

# Low-pT GSF electrons (PreId unbiased)
print 
plot( string="Low pT GSF trk (PreId), AxE",
      plt=plt, draw_roc=True, draw_eff=False,
      df=test, selection=has_gsf, discriminator=test.gsf_bdtout1, mask=None,
      label='Low-$p_{T}$ GSF track + unbiased ($\mathcal{A}\epsilon$)',
      color='red', markersize=8, linestyle='dashed',
)

# Low-pT GSF electrons (CMSSW)
print 
plot( string="Low pT ele (CMSSW), AxE",
      plt=plt, draw_roc=True, draw_eff=False,
      df=test, selection=has_ele, discriminator=test.ele_mva_value, mask=None,
      label='Low-$p_{T}$ ele + 2019Jun28 model ($\mathcal{A}\epsilon$)',
      color='blue', markersize=8, linestyle='dashdot',
)

# Low-pT GSF electrons (retraining)
print 
eff2,fr2,roc2 = plot( string="Low pT ele (latest), AxE",
                      plt=plt, draw_roc=True, draw_eff=False,
                      df=test, selection=has_ele, discriminator=test.training_out, mask=None,
                      label='Low-$p_{T}$ ele + latest model ($\mathcal{A}\epsilon$)',
                      color='blue', markersize=8, linestyle='solid',
)

roc = (roc2[0]*fr2,roc2[1]*eff2,roc2[2]) 
idxL = np.abs(roc[0]-fr1).argmin()
idxT = np.abs(roc[1]-eff1).argmin()
print "   PFele: eff/fr/thresh:","{:.3f}/{:.4f}/{:4.2f} ".format(eff1,fr1,np.nan)
print "   Loose: eff/fr/thresh:","{:.3f}/{:.4f}/{:4.2f} ".format(roc[1][idxL],roc[0][idxL],roc[2][idxL])
print "   Tight: eff/fr/thresh:","{:.3f}/{:.4f}/{:4.2f} ".format(roc[1][idxT],roc[0][idxT],roc[2][idxT])

# Adapt legend
def update_prop(handle, orig):
   handle.update_from(orig)
   handle.set_marker("o")
plt.legend(handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)})

plt.xlabel('Mistag rate')
plt.ylabel(r'Acceptance $\times$ efficiency')
plt.legend(loc='best')
#plt.legend(loc='lower left', bbox_to_anchor=(0., 1.1)) 
plt.xlim(0., 1)
plt.gca().set_xscale('log')
plt.xlim(1e-4, 1)

try : plt.savefig('%s/output/roc.png'%(base))
except : print 'Issue: %s/output/roc.png'%(base)
try : plt.savefig('%s/output/roc.pdf'%(base))
except : print 'Issue: %s/output/roc.pdf'%(base)
plt.clf()
