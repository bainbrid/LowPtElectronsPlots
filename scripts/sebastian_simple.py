################################################################################
# Imports ...

from __future__ import print_function
import builtins
import future
from future.utils import raise_with_traceback

import uproot
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
import matplotlib.pyplot as plt

import ROOT as r 
from setTDRStyle import setTDRStyle

import pickle

################################################################################
# I/O

# Sebastian, files to use:
# /eos/user/b/bainbrid/lowpteleid/nonres_large/output_0.root # small
# /eos/user/b/bainbrid/lowpteleid/nonres_large/output_000.root # large

# Input files
filenames = [
    #"/eos/user/b/bainbrid/lowpteleid/nonres_large/output_000.root",
    #"../data/170823/nonres_large/output_0.root",   # Rob's small local version
    #"../data/170823/nonres_large/output_000.root", # Rob's large local version
    #"output/output_230816.root",
    #"output/output_231011.root",
    #"output/output_old.root",
    #"output/output_new.root",
    #"output/output.root",
    "output/output_simple.root",
    ]
    
columns = [
    'is_e','is_egamma','has_ele', # LABELING
    'tag_pt','tag_eta','ele_pt','ele_eta', # KINE
    'ele_mva_value','ele_mva_value_retrained', # ID SCORES
    'evt','weight','rho', # MISC
    ]
columns = list(set(columns))

# Extract branches from root file as a pandas data frame
dfs = [ uproot.open(i)['ntuplizer/tree'].arrays(columns,library="pd")  for i in filenames ]
df = pd.concat(dfs)
print(df.describe(include='all').T)

################################################################################
# Filters applied to branches
################################################################################

# Filter data frame based on tag-side muon pT and eta
tag_muon_pt = 5.0
tag_muon_eta = 1.5
df = df[ (df.tag_pt>tag_muon_pt) & (np.abs(df.tag_eta)<tag_muon_eta) ]

# Filter data frame for PF electrons (removing low-pT content)
egamma = df[df.is_egamma]

# Filter data frame keeping only PF electron candidates (removing tracks, etc)
eta_upper = 2.5
pt_lower = 2.0
pt_upper = 1.e6
#has_pfele = (egamma.has_ele) & (np.abs(egamma.ele_eta)<eta_upper) & (egamma.ele_pt>pt_lower)
has_pfele  = ( (egamma.has_ele) & (np.abs(egamma.ele_eta)<2.5) )
has_pfele &= ( (egamma.ele_pt>pt_lower) & (egamma.ele_pt<pt_upper) )

print(pd.crosstab(
    egamma.is_e,
    [has_pfele],
    rownames=['is_e'],
    colnames=['has_pfele'],
    margins=True))

################################################################################
# ROC curve using matplotlib
################################################################################

plt.figure(figsize=(6,6))
ax = plt.subplot(111)
plt.title('PF electron performance')

xmin = 1.e-4
plt.xlim(xmin,1.)
plt.ylim([0., 1.])

plt.xlabel(f"Mistag rate (w.r.t. reco'ed PF ele cand, pT > {pt_lower:.1f} GeV, |eta| < {eta_upper})")
plt.ylabel(f"Efficiency (w.r.t. PF ele cand matched to truth, pT > {pt_lower:.1f} GeV, |eta| < {eta_upper})")
ax.tick_params(axis='x', pad=10.)
plt.grid(True)
plt.gca().set_xscale('log')

# "by chance" line
plt.plot(
    np.arange(xmin,1.,xmin),
    np.arange(xmin,1.,xmin),
    ls='dotted',
    lw=0.5,
    label="By chance"
    )

# PF ID (default)
pf_id_branch = 'ele_mva_value'
pf_id_fpr,pf_id_tpr,pf_id_thr = roc_curve(
    egamma.is_e[has_pfele],
    egamma[pf_id_branch][has_pfele]
    )
pf_id_auc = roc_auc_score(
    egamma.is_e[has_pfele],
    egamma[pf_id_branch][has_pfele]
    )
plt.plot(
    pf_id_fpr,
    pf_id_tpr,
    linestyle='dotted', color='purple', linewidth=1.0,
    label='PF default ID ({:.3f})'.format(pf_id_auc)
    )

# PF ID (retrained)
pf_id_retrain_branch = 'ele_mva_value_retrained'
pf_id_retrain_fpr,pf_id_retrain_tpr,pf_id_retrain_thr = roc_curve(
    egamma.is_e[has_pfele],
    egamma[pf_id_retrain_branch][has_pfele]
    )
pf_id_retrain_auc = roc_auc_score(
    egamma.is_e[has_pfele],
    egamma[pf_id_retrain_branch][has_pfele]
    )
plt.plot(
    pf_id_retrain_fpr, 
    pf_id_retrain_tpr,
    linestyle='solid', color='purple', linewidth=1.0,
    label='PF retrained ID ({:.3f})'.format(pf_id_retrain_auc)
    )

# EXPORT PICKLE FILE WITH ROCS FROM SEBASTIAN SIMPLE SCRIPT !!!
data = (
    pf_id_retrain_fpr,pf_id_retrain_tpr,pf_id_retrain_thr,pf_id_retrain_auc, # Retrained
    pf_id_fpr,pf_id_tpr,pf_id_thr,pf_id_auc # Original
    )
f = open('id_pf_retrain.pkl','wb')
pickle.dump(data,f)

plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
plt.tight_layout()
plt.savefig('./roc_simple.pdf')
plt.clf()
plt.close()
print('Created file: ./roc_simple.pdf')

