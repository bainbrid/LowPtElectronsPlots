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

################################################################################
# I/O

# Sebastian, files to use:
# /eos/user/b/bainbrid/lowpteleid/nonres_large/output_0.root # small
# /eos/user/b/bainbrid/lowpteleid/nonres_large/output_000.root # large

# Input files
filenames = [
    #"/eos/user/b/bainbrid/lowpteleid/nonres_large/output_000.root",
    "../data/170823/nonres_large/output_000.root", # Rob's local version
    ]
    
columns = [
    'is_e','is_egamma','has_ele', # LABELING
    'tag_pt','tag_eta','ele_pt','ele_eta', # KINE
    'ele_mva_value','ele_mva_value_retrained', # ID SCORES
    'evt','weight','rho', # MISC
    ]
columns = list(set(columns))

# Extract branches from root file as a pandas data frame
df = [ uproot.open(i)['ntuplizer/tree'].arrays(columns,library="pd")  for i in filenames ]
data = pd.concat(df)

################################################################################
# Filters applied to branches
################################################################################

# Filter data frame based on tag-side muon pT and eta
tag_muon_pt = 7.0
tag_muon_eta = 1.5
data = data[ (data.tag_pt>tag_muon_pt) & (np.abs(data.tag_eta)<tag_muon_eta) ]

# Filter data frame for PF electrons (removing low-pT content)
egamma = data[data.is_egamma]

# Filter data frame keeping only PF electron candidates (removing tracks, etc)
pt_lower = 2.0
pt_upper = 5.0
has_pfele = (egamma.has_ele) & (np.abs(egamma.ele_eta)<2.5) & (egamma.ele_pt>pt_lower)
egamma = egamma[has_pfele]

################################################################################
# ROC curve using matplotlib
################################################################################

plt.figure(figsize=(6,6))
ax = plt.subplot(111)
plt.title('PF electron performance')

xmin = 1.e-4
plt.xlim(xmin,1.)
plt.ylim([0., 1.])

plt.xlabel(f"Mistag rate (w.r.t. reco'ed PF ele cand, pT > {pt_lower:.1f} GeV)")
plt.ylabel(f"Efficiency (w.r.t. PF ele cand matched to truth, pT > {pt_lower:.1f} GeV)")
ax.tick_params(axis='x', pad=10.)
plt.grid(True)
#plt.gca().set_xscale('log')

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

plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
plt.tight_layout()
plt.savefig('./roc_simple.pdf')
plt.clf()
plt.close()
print('Created file: ./roc_simple.pdf')
