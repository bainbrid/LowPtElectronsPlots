import numpy as np
import pandas as pd

import ROOT
import uproot
import rootpy

import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties

from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb

import json
import os
#from pdb import set_trace
#set_trace()

#########

print "### Define parameters ..."

tag = '2019Oct10'
dataset = 'test'
what = 'cmssw_mva_id_extended'

input_files = {
   'test':['/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15/src/2-ntuples-from-crab/output.CURRENT.root']
}

base='/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15/src/LowPtElectrons/LowPtElectrons/macros'

mods=base+'/models/'+tag
if not os.path.isdir(mods) : os.makedirs(mods)

plots=base+'/plots/'+tag
if not os.path.isdir(plots) : os.makedirs(plots)

features=[
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
   'gsf_bdtout1',
]

additional=[
   'ele_mva_value',
   'ele_mva_id',
   'ele_pt',
   'gsf_bdtout2',
]

labeling=[
   'is_e', 
   'is_e_not_matched',
   'is_other',
   'is_egamma',
   'has_trk',
   'has_seed',
   'has_gsf',
   'has_pfgsf',
   'has_ele',
   'seed_trk_driven',
   'seed_ecal_driven'
]
fields=features+additional+labeling

fields+=['gsf_pt']
fields=list(set(fields))

fields+=[
   'gen_pt', 
   'gen_eta', 
   'trk_pt', 
   'trk_eta', 
   'trk_charge', 
   'trk_dr',
   'seed_trk_driven', 
   'seed_ecal_driven',
   'gsf_pt', 
   'gsf_eta', 
   'gsf_dr', 
   'ele_dr',
   'pfgsf_pt', 
   'pfgsf_eta',
   'ele_pt', 
   'ele_eta', 
   'ele_dr',
   'evt',
   'weight'
]
fields=list(set(fields))

#data_dict = get_data_sync(dataset, features) # path='features/tree')
#def get_data_sync(dataset, columns, nthreads=2*multiprocessing.cpu_count(), exclude={}, path='ntuplizer/tree'):

print "### Load files ..."

if dataset not in input_files:
   raise ValueError('The dataset %s does not exist, I have %s' % (dataset, ', '.join(input_files.keys())))

print 'getting files from "%s": ' % dataset
print ' \n'.join(input_files[dataset])
infiles = [ uproot.open(i) for i in input_files[dataset] ]

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
   for column in columns:
      ret[column] = np.concatenate((ret[column],arrays[column]))
data_dict = ret

print "### Pre-process data ..."

if 'is_e_not_matched' not in data_dict:
   data_dict['is_e_not_matched'] = np.zeros(data_dict['trk_pt'].shape, dtype=bool)

data = pd.DataFrame(data_dict)
#data = data.head(1000)

data['prescale'] = data.weight
data['weight'] = np.ones(data.weight.shape)
data = data[np.invert(data.is_e_not_matched)] 

data['training_out'] = -1.
log_trkpt = np.log10(data.trk_pt)
log_trkpt[np.isnan(log_trkpt)] = -9999
data['log_trkpt'] = log_trkpt

print "### Load binning and weights ..."

kmeans_model = '%s/kmeans_%s_weighter.pkl'%(mods, dataset)
if not os.path.isfile(kmeans_model):
   raise ValueError('I could not find the appropriate model, I have %s/kmeans_%s_weighter.pkl' % (mods, dataset))

apply_weight = np.vectorize(lambda x, y: y.get(x), excluded={2})
kmeans = joblib.load(kmeans_model)
cluster = kmeans.predict(data[['log_trkpt', 'trk_eta']])
str_weights = json.load(open(kmeans_model.replace('.pkl', '.json')))
weights = {}
for i in str_weights:
   try:
      weights[int(i)] = str_weights[i]
   except:
      pass
weights = apply_weight(cluster, weights)
data['weight'] = weights*np.invert(data.is_e) + data.is_e

print data.describe()
print data.info()

if 'trk_charge' in data.columns:
   for feat in ['ktf_ecal_cluster_dphi', 'ktf_hcal_cluster_dphi', 'preid_trk_ecal_Dphi']:
      if feat in data.columns:
         data[feat] = data[feat]*data['trk_charge']

for c in data.columns:
   if data[c].dtype == np.dtype('bool') and c not in labeling:
      data[c] = data[c].astype(int) # convert bools to integers

egamma = data[data.is_egamma]          # EGamma electrons
orig = data.copy()                     # all electrons
data = data[np.invert(data.is_egamma)] # low pT electrons
print "orig.shape",orig.shape
print "lowpt.shape",data.shape
print "egamma.shape",egamma.shape

print "### Define train/test/validation data sets ..."

def train_test_split(data, div, thr):
   mask = data.evt % div
   mask = mask < thr
   return data[mask], data[np.invert(mask)]

train_test, validation = train_test_split(data, 10, 8)
train, test = train_test_split(train_test, 10, 6)

print "### Loading model ..."

full_model = '%s/bdt_%s/%s__%s_BDT.pkl' % (mods, what, dataset, what)
clf = joblib.load(full_model)
training_out = clf.predict_proba(validation[features].as_matrix())[:, 1]
validation['training_out'] = training_out

print "### Plotting ..."

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--what',default='cmssw_mva_id_extended',type=str)
parser.add_argument('--jobtag',default='',type=str)
args = parser.parse_args()

from plotting import *
plotting(plots,dataset,args,validation,egamma,data)

#################################################################################
#
#def plot( plt, df, string, selection, draw_roc, draw_eff, 
#          label, color, markersize, linestyle, linewidth=1.0, discriminator=None, mask=None, 
#          df_xaxis=None ) :
#   
#   if draw_roc is True and discriminator is None : 
#      print "No discriminator given for ROC curve!"
#      quit()
#   print string
#   if mask is None : mask = [True]*df.shape[0]
#   denom = df.is_e#[mask]; 
#   numer = denom & selection#[mask]
#   eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   print "   eff/numer/denom: {:6.4f}".format(eff), numer.sum(), denom.sum()
#   denom = ~df.is_e[mask]; numer = denom & selection[mask]
#   if df_xaxis is not None : denom = ~df_xaxis.is_e # change x-axis denominator!
#   mistag = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   print "    fr/numer/denom: {:6.4f}".format(mistag), numer.sum(), denom.sum()
#   if draw_roc :
#      roc = roc_curve(df.is_e[selection&mask], discriminator[selection&mask])
#      auc = roc_auc_score(df.is_e[selection&mask], discriminator[selection&mask])
#      plt.plot(roc[0]*mistag, roc[1]*eff, 
#               linestyle=linestyle,
#               linewidth=linewidth,
#               color=color,
#               label=label+', AUC: %.3f'%auc)
#      plt.plot([mistag], [eff], marker='o', color=color, markersize=markersize)
#      return eff,mistag,roc
#   elif draw_eff :
#      plt.plot([mistag], [eff], marker='o', color=color, markersize=markersize, 
#               label=label)
#      return eff,mistag,None
#
#################################################################################
#
#def AxE_retraining(plt,df_lowpt,df_egamma) :
#
#   print 
#   print "AxE_retraining"
#   
#   # Low-pT GSF electrons (PreId unbiased)
#   has_gsf = (df_lowpt.has_gsf) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
#   plot( plt=plt, df=df_lowpt, string="Low pT GSF trk (PreId), AxE",
#         selection=has_gsf, draw_roc=True, draw_eff=False,
#         label='Low-$p_{T}$ GSF track + unbiased ($\mathcal{A}\epsilon$)',
#         color='red', markersize=8, linestyle='dashed',
#         discriminator=df_lowpt.gsf_bdtout1,
#   )
#   
#   # Low-pT GSF electrons (CMSSW)
#   has_gsf = (df_lowpt.has_gsf) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
#   has_ele = (df_lowpt.has_ele) & (df_lowpt.ele_pt>0.5) & (np.abs(df_lowpt.ele_eta)<2.4)
#   plot( plt=plt, df=df_lowpt, string="Low pT ele (CMSSW), AxE",
#         selection=has_ele, draw_roc=True, draw_eff=False,
#         label='Low-$p_{T}$ ele + 2019Jun28 model ($\mathcal{A}\epsilon$)',
#         color='blue', markersize=8, linestyle='dashdot',
#         discriminator=df_lowpt.ele_mva_value,
#   )
#   
#   # Low-pT GSF electrons (retraining)
#   has_gsf = (df_lowpt.has_gsf) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
#   has_ele = (df_lowpt.has_ele) & (df_lowpt.ele_pt>0.5) & (np.abs(df_lowpt.ele_eta)<2.4)
#   eff2,fr2,roc2 = plot( plt=plt, df=df_lowpt, string="Low pT ele (latest), AxE",
#                         selection=has_ele, draw_roc=True, draw_eff=False,
#                         label='Low-$p_{T}$ ele + latest model ($\mathcal{A}\epsilon$)',
#                         color='blue', markersize=8, linestyle='solid',
#                         discriminator=df_lowpt.training_out,
#   )
#   
#   # EGamma PF GSF track 
#   has_gsf = (df_egamma.has_gsf) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
#   has_pfgsf = (df_egamma.has_pfgsf) & (df_egamma.pfgsf_pt>0.5) & (np.abs(df_egamma.pfgsf_eta)<2.4)
#   eff1,fr1,_ = plot( plt=plt, df=df_egamma, string="EGamma GSF trk, AxE",
#                      selection=has_pfgsf, draw_roc=False, draw_eff=True,
#                      label='EGamma GSF track ($\mathcal{A}\epsilon$)',
#                      color='green', markersize=8, linestyle='solid',
#                      mask = has_gsf,
#   )
#   
#   # EGamma PF ele 
#   has_gsf = (df_egamma.has_gsf) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
#   has_ele = (df_egamma.has_ele) & (df_egamma.ele_pt>0.5) & (np.abs(df_egamma.ele_eta)<2.4)
#   plot( plt=plt, df=df_egamma, string="EGamma PF ele, AxE",
#         selection=has_ele, draw_roc=False, draw_eff=True,
#         label='EGamma PF ele ($\mathcal{A}\epsilon$)',
#         color='purple', markersize=8, linestyle='solid',
#         mask = has_gsf,
#   )
#
#   roc = (roc2[0]*fr2,roc2[1]*eff2,roc2[2]) 
#   idxL = np.abs(roc[0]-fr1).argmin()
#   idxT = np.abs(roc[1]-eff1).argmin()
#   print "   PFele: eff/fr/thresh:",\
#      "{:.3f}/{:.4f}/{:4.2f} ".format(eff1,fr1,np.nan)
#   print "   Loose: eff/fr/thresh:",\
#      "{:.3f}/{:.4f}/{:4.2f} ".format(roc[1][idxL],roc[0][idxL],roc[2][idxL])
#   print "   Tight: eff/fr/thresh:",\
#      "{:.3f}/{:.4f}/{:4.2f} ".format(roc[1][idxT],roc[0][idxT],roc[2][idxT])
#
#################################################################################
#
#def plotting(plots,dataset,what,jobtag,df_lowpt,df_egamma,df_orig) :
#   print "plotting() ..."
#
#   plots_list = [
#      {"method":AxE_retraining,"args":(plt,df_lowpt,df_egamma),"suffix":"AxE_retraining",},
#      #{"method":effs_wrt_gsf_tracks,"args":(plt,df_lowpt,df_egamma),"suffix":"effs_wrt_gsf_tracks",},
#      ]
#   
#   for plot in plots_list :
#
#      plt.figure(figsize=[8, 12])
#      ax = plt.subplot(111)
#      box = ax.get_position()
#      ax.set_position([box.x0, box.y0, box.width, box.height*0.666])
#      plt.title('%s training' % what.replace("_"," "))
#      plt.plot(np.arange(0.,1.,0.01),np.arange(0.,1.,0.01),'k--')
#
#      plot["method"](*plot["args"]) # Execute method
# 
#      # Adapt legend
#      def update_prop(handle, orig):
#         handle.update_from(orig)
#         handle.set_marker("o")
#      plt.legend(handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)})
#
#      plt.xlabel('Mistag Rate')
#      plt.ylabel('Efficiency')
#      plt.legend(loc='lower left', bbox_to_anchor=(0., 1.1)) #plt.legend(loc='best')
#      plt.xlim(0., 1)
#      
#      tupl = (plots, dataset, jobtag, what, plot["suffix"])
#      #try : plt.savefig('%s/%s_%s_%s_BDT_%s.png' % (tupl))
#      #except : print 'Issue: %s/%s_%s_%s_BDT_%s.png' % (tupl)
#      #try : plt.savefig('%s/%s_%s_%s_BDT_%s.pdf' % (tupl))
#      #except : print 'Issue: %s/%s_%s_%s_BDT_%s.pdf' % (tupl)
#      plt.gca().set_xscale('log')
#      plt.xlim(1e-4, 1)
#      try : plt.savefig('%s/%s_%s_%s_log_BDT_%s.png' % (tupl))
#      except : print 'Issue: %s/%s_%s_%s_log_BDT_%s.png' % (tupl)
#      try : plt.savefig('%s/%s_%s_%s_log_BDT_%s.pdf' % (tupl))
#      except : print 'Issue: %s/%s_%s_%s_log_BDT_%s.pdf' % (tupl)
#      
#      plt.clf()
#
#################################################################################
#
#plotting(plots,dataset,what,"",validation,egamma,data)
