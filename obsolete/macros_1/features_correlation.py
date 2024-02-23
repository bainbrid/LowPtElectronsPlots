import numpy as np
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
from pdb import set_trace

import matplotlib.pyplot as plt
import uproot
import json
import pandas as pd
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
from datasets import tag, pre_process_data, target_dataset
import os

dataset = 'test' #target_dataset

mods = '%s/src/LowPtElectrons/LowPtElectrons/macros/models/%s/' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(mods):
   os.makedirs(mods)

plots = '%s/src/LowPtElectrons/LowPtElectrons/macros/plots/%s/feature_selection/' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(plots):
   os.makedirs(plots)

from features import *
features, additional = get_features('combined_id')
fields = features+labeling+additional

data = pre_process_data(dataset, fields, False)
electrons = data[data.is_e & (np.random.rand(data.shape[0]) < 0.3)]
tracks = data[data.is_other & (np.random.rand(data.shape[0]) < 0.1)]

features.pop(features.index('trk_high_purity'))

plt.clf()
plt.figure(figsize=[11,11])
corrmat = data[features].corr(method='pearson', min_periods=1)
heatmap = plt.pcolor(corrmat, cmap = "RdBu", vmin=-1, vmax=+1)
plt.colorbar()#heatmap1, ax=ax1)

plt.xlim(0, len(features)+1)
plt.ylim(0, len(features)+1)

plt.gca().set_xticks(np.arange(len(features))+0.5, minor=False)
plt.gca().set_yticks(np.arange(len(features))+0.5, minor=False)
xlabels = plt.gca().set_xticklabels(features, minor=False, ha='right', rotation=90)
ylabels = plt.gca().set_yticklabels(features, minor=False)
[i.set_fontsize(9) for i in xlabels]
[i.set_fontsize(9) for i in ylabels]

plt.tight_layout()
plt.savefig('%s/correlation.png' % plots)
plt.savefig('%s/correlation.pdf' % plots)
plt.clf()

from itertools import combinations
hyper_corr = set()
pairs = []
for f1, f2 in combinations(features, 2):
   corr = corrmat[f1][f2]
   if abs(corr) > 0.95:
      pairs.append((f1, f2))
      hyper_corr.add(f1)
      hyper_corr.add(f2)

hyper_corr = list(hyper_corr)
corrmat = data[hyper_corr].corr(method='pearson', min_periods=1)
heatmap = plt.pcolor(corrmat, cmap = "RdBu", vmin=-1, vmax=+1)
plt.colorbar()#heatmap1, ax=ax1)

plt.xlim(0, len(hyper_corr)+1)
plt.ylim(0, len(hyper_corr)+1)

plt.gca().set_xticks(np.arange(len(hyper_corr))+0.5, minor=False)
plt.gca().set_yticks(np.arange(len(hyper_corr))+0.5, minor=False)
xlabels = plt.gca().set_xticklabels(hyper_corr, minor=False, ha='right', rotation=90)
ylabels = plt.gca().set_yticklabels(hyper_corr, minor=False)
[i.set_fontsize(9) for i in xlabels]
[i.set_fontsize(9) for i in ylabels]

plt.tight_layout()
plt.savefig('%s/large_correlation.png' % plots)
plt.savefig('%s/large_correlation.pdf' % plots)
plt.clf()

combined_importances = [i.split()[0] for i in open('models/2018Oct05/bdt_bo_combined_id/importances.txt')]
to_drop = set()
for f1, f2 in pairs:
   plt.clf()
   plt.scatter(electrons[f1], electrons[f2], label='electrons', c='b')
   plt.scatter(tracks[f1], tracks[f2], label='tracks', c='r')
   plt.legend(loc='best')
   plt.xlabel(f1)
   plt.ylabel(f2)
   plt.savefig('%s/%s_vs_%s.png' % (plots, f1, f2))
   plt.savefig('%s/%s_vs_%s.pdf' % (plots, f1, f2))
   plt.clf()
   idx1 = combined_importances.index(f1)
   idx2 = combined_importances.index(f2)
   if idx1 > idx2:
      print f2,'>',f1
      to_drop.add(f1)
   else:
      print f1,'>',f2
      to_drop.add(f2)

print 'You can drop'
print to_drop
