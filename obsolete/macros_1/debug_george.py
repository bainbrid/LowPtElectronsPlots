import numpy as np
import matplotlib
matplotlib.use('Agg')
from pdb import set_trace
import os
from glob import glob
import pandas as pd
import json
from pprint import pprint
import matplotlib.pyplot as plt

from features import *
biased_features, additional = get_features('cmssw_displaced_improvedfullseeding')
unbiased_features, additional = get_features('cmssw_improvedfullseeding')

import uproot
import datasets as dsets
old_selection  = dsets.training_selection
dsets.training_selection = lambda x: np.ones(x.shape[0], dtype=bool)

to_dump = [
   'run',
   'lumi',
   'evt',
   'gen_pt',
   'gen_eta',
   'gen_phi',
   'trk_pt',
   'trk_eta',
   'trk_phi',
   'gsf_pt',
   'gsf_eta',
   'gsf_phi',
   ]

dsets.input_files['debug'] = ['/afs/cern.ch/work/m/mverzett/RK102v3/src/LowPtElectrons/LowPtElectrons/run/george_synch_all.root']
#fields = set(biased_features+additional+unbiased_features+to_dump+labeling)
fields = set(biased_features+unbiased_features+to_dump+labeling)
data = dsets.pre_process_data(
   'debug', list(fields),
   for_seeding=False,
   keep_nonmatch=True
)

from sklearn.externals import joblib
import xgboost as xgb

biased_model = joblib.load('/afs/cern.ch/work/m/mverzett/RecoEgamma-ElectronIdentification/LowPtElectrons/RunII_Fall17_LowPtElectrons_displaced_pt_eta_biased.pkl')
biased_model.booster = lambda : biased_model._Booster
#'models/2018Nov01/bdt_bo_displaced_improvedfullseeding_noweight/model_18.pkl')
unbiased_model = joblib.load('/afs/cern.ch/work/m/mverzett/RecoEgamma-ElectronIdentification/LowPtElectrons/RunII_Fall17_LowPtElectrons_unbiased.pkl')
unbiased_model.booster = lambda :unbiased_model._Booster

##def _monkey_patch():
##    return model._Booster
##
##for model in [biased_model, unbiased_model]:
##    if isinstance(model.booster, basestring):
##        model.booster = _monkey_patch

biased_out = biased_model.predict_proba(data[biased_features].values)[:,1]
biased_out[np.isnan(biased_out)] = -999 #happens rarely, but happens

unbiased_out = unbiased_model.predict_proba(data[unbiased_features].values)[:,1]
unbiased_out[np.isnan(unbiased_out)] = -999 #happens rarely, but happens

data['biased_out'] = biased_out
data['unbiased_out'] = unbiased_out
data['gen_match_george'] = False
data['trk_match_george'] = False

to_dump.append('unbiased_out')
to_dump.append('biased_out')
data[data.is_e][to_dump].to_csv('/afs/cern.ch/work/m/mverzett/public/george_sync_10_2.csv', index=False)
limited = data[data.is_e][to_dump]
raw = dsets.get_data_sync('debug', list(fields-{'trk_dxy_sig'}))
raw = pd.DataFrame(raw)
#raw = raw[raw.is_e]
passed = set(zip(limited.lumi, limited.evt))
raw_passed = set(zip(raw.lumi, raw.evt))

wp = {
   'biased' : {
      "L" : -0.48,
      "M" : 0.76,
      "T" : 1.83,
      },
   'unbiased' : {
      "L" : 1.03,
      "M" : 1.75,
      "T" : 2.61,
      },
}

electrons = data[data.is_e].copy()
electrons['matched_to'] = -1
nelectrons = float(electrons.shape[0])
print 'Total electrons: %d' % electrons.shape[0]
print 'Track efficiency: %.3f' % ((electrons.trk_pt > 0).sum()/nelectrons)
print 'biased L efficiency: %.3f' % ((electrons.biased_out > wp['biased']['L']).sum()/nelectrons)
print 'biased M efficiency: %.3f' % ((electrons.biased_out > wp['biased']['M']).sum()/nelectrons)
print 'biased T efficiency: %.3f' % ((electrons.biased_out > wp['biased']['T']).sum()/nelectrons)
print 'unbiased L efficiency: %.3f' % ((electrons.unbiased_out > wp['unbiased']['L']).sum()/nelectrons)
print 'unbiased M efficiency: %.3f' % ((electrons.unbiased_out > wp['unbiased']['M']).sum()/nelectrons)
print 'unbiased T efficiency: %.3f' % ((electrons.unbiased_out > wp['unbiased']['T']).sum()/nelectrons)
## print 'Track-based GSF efficiency: %3f' % ((electrons.baseline > 0.5).sum()/nelectrons)

delta_R = lambda entry: np.sqrt((entry.gen_eta - entry.trk_eta)**2 + (entry.gen_phi - entry.trk_phi)**2)

george = pd.read_csv('/afs/cern.ch/user/g/gkaratha/public/debug.csv')

print '\n\n\nGeorge Results \n'
ngeorge = float(george.shape[0])
print 'Total electrons: %d' % george.shape[0]
print 'Track efficiency: %.3f' % ((george.trk_pt > 0).sum()/ngeorge)
print 'biased L efficiency: %.3f' % ((george.biased_out > wp['biased']['L']).sum()/ngeorge)
print 'biased M efficiency: %.3f' % ((george.biased_out > wp['biased']['M']).sum()/ngeorge)
print 'biased T efficiency: %.3f' % ((george.biased_out > wp['biased']['T']).sum()/ngeorge)
print 'unbiased L efficiency: %.3f' % ((george.unbiased_out > wp['unbiased']['L']).sum()/ngeorge)
print 'unbiased M efficiency: %.3f' % ((george.unbiased_out > wp['unbiased']['M']).sum()/ngeorge)
print 'unbiased T efficiency: %.3f' % ((george.unbiased_out > wp['unbiased']['T']).sum()/ngeorge)

delta_r = np.vectorize(lambda e, p, e2, p2: np.sqrt((e-e2)**2+(p-p2)**2))


def almost_equal(v1, v2):
   return abs(v1 - v2)/abs(v2)  < 10**-3

def almost_same(e1, e2, val='gen'):
   return almost_equal(e1['%s_pt' % val], e2['%s_pt' % val]) and \
      almost_equal(e1['%s_eta' % val], e2['%s_eta' % val]) and \
      almost_equal(e1['%s_phi' % val], e2['%s_phi' % val])

from itertools import product

from pdb import set_trace
george['matched_to'] = -1
same_gen = []
same_trk = []
same_gsf = []
same_biased = []
same_unbias = []
mached_to = []

for idx, entry in electrons.iterrows():
   same_gen.append(False)
   same_trk.append(False)
   same_gsf.append(False)
   same_biased.append(-999)
   same_unbias.append(-999)
   mached_to.append(-1)
   lumi, evt = entry.lumi, entry.evt
   geor = george[(george.lumi == lumi) & (george.evt == evt)]    
   for george_idx, e2 in geor.iterrows():
      if almost_same(entry, e2, 'gen'):
         same_gen[-1] = True
         george['matched_to'].loc[george_idx] = idx
         mached_to[-1] = george_idx
         entry.matched_to = george_idx
         if almost_same(entry, e2, 'trk'):
            same_trk[-1] = True
            if almost_same(entry, e2, 'gsf'):
               same_gsf[-1] = True
               same_biased[-1] = e2.biased_out
               same_unbias[-1] = e2.unbiased_out
     
electrons['same_gen'] = same_gen
electrons['same_trk'] = same_trk
electrons['same_gsf'] = same_gsf
electrons['george_bias'] = same_biased
electrons['george_unbias'] = same_unbias
electrons['matched_to'] = mached_to

#failing = (electrons.trk_pt > 0) & np.invert(electrons['bdt_match_pass'])
same_gsf = electrons[electrons.same_gsf & (electrons.gsf_pt > 0)][['george_bias', 'george_unbias', 'biased_out', 'unbiased_out']]

not_same_gsf = electrons[electrons.same_trk & np.invert(electrons.same_gsf)]['matched_to']

def compare(e1, e2):
  for val in to_dump:
    print ('%20s' % val),'\t',e1[val], '  -->  ',e2[val]

for i, j in zip(not_same_gsf.keys(), not_same_gsf):
  compare(electrons.loc[i], george.loc[j])
  print '\n'
