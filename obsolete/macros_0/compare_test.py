
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import ROOT
import uproot
import rootpy
import json
import pandas as pd
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from datasets import tag, pre_process_data
import os

mods = '%s/src/LowPtElectrons/LowPtElectrons/macros/models/%s/' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(mods):
   os.mkdirs(mods)

plots = 'sip_checks'
if not os.path.isdir(plots):
   os.makedirs(plots)


## new_id   = json.loads(open('models/2018Oct22v2/bdt_bo_id/roc.json').read())
## mva_like = json.loads(open('models/2018Oct22v2/bdt_bo_mva_id/roc.json').read())
## combined = json.loads(open('models/2018Oct22v2/bdt_bo_combined_id/roc.json').read())
## seedingOct05 = json.loads(open('models/2018Oct05/bdt_bo_seeding/roc.json').read())
## seeding3 = json.loads(open('models/2018Oct05/bdt_bo_seeding_2018Oct22v2Model/roc.json').read())
## improvedseeding = json.loads(open('models/2018Oct22v2/bdt_bo_improvedseeding/roc.json').read())
## seeding_nw = json.loads(open('models/2018Oct22v2/bdt_bo_seeding_noweight/roc.json').read())
## full_seeding = json.loads(open('models/2018Oct22v2/bdt_bo_fullseeding/roc.json').read())
## old = json.loads(open('plots/2018Oct22v2/BToKee__combined_id_ROCS.json').read())

ids = [
   (json.loads(open('models/2019Feb05/bdt_cmssw_mva_id/roc.json').read()), 'previous RECO settings + OLD model', False),
   (json.loads(open('models/2019Feb22/id_2019Feb05/roc.json').read()), 'What is currently in CMSSW', False),
   (json.loads(open('models/2019Feb22/bdt_cmssw_mva_id/roc.json').read()), 'retraining', False),
]

seedings = [
   ## (json.loads(open('models/2018Nov01/bdt_bo_seeding/roc.json').read()), 'seeding', False),
   ## (json.loads(open('models/2018Nov01/bdt_bo_trkonly/roc.json').read()), 'TRK Only', False),
   ## (json.loads(open('models/2018Nov01/bdt_bo_fullseeding/roc.json').read()), 'seeding + simple GSF', False),
   ## (json.loads(open('models/2018Nov01/bdt_bo_betterseeding/roc.json').read()), 'seeding + rho', False),
   ## (json.loads(open('models/2018Nov01/bdt_bo_improvedseeding/roc.json').read()), 'seeding + ECAL + HCAL', True),

   ## (json.loads(open('models/2018Nov01/bdt_bo_displaced_improvedfullseeding_noweight/roc.json').read()), '9.4.X biased model on 9.4.X simulation', False),
   ## (json.loads(open('models/2019Feb05/cmssw_biased_bdt/roc.json').read()), '9.4.X biased model on 10.2.X simulation', False),
   ## (json.loads(open('models/2019Feb05/cmssw_biased_bdt_myvars/roc.json').read()), 'myvars', False),
   ## (json.loads(open('models/2019Jan30CMSSW102X/cmssw_biased_bdt_with_nomatch/roc.json').read()), 'old', False),
   ## (json.loads(open('models/2019Jan26Selected/94XTraining_displaced_improvedfullseeding/roc.json').read()), 'older', False),
   ## (json.loads(open('models/2019Jan26Selected/94XTraining_try2/roc.json').read()), 'older V2', False),
   ## (json.loads(open('models/2019Jan26Selected/94XTraining_myvars/roc.json').read()), 'older myvars', False),
   (json.loads(open('models/2019Feb05/bdt_cmssw_displaced_improvedfullseeding/roc.json').read()), '10.2.X biased model, 9.4.X Hyperparameters', False),
   ## (json.loads(open('models/2019Feb05/bdt_bo_cmssw_displaced_improvedfullseeding_noweight/roc.json').read()), '10.2.X biased model, new Bayesian Optimisation', False),
   (json.loads(open('models/2019Feb05/bdt_cmssw_displaced_improvedfullseeding_fixSIP/roc.json').read()), '10.2.X biased model, correct IP', False),
   (json.loads(open('models/2019Feb05/bdt_cmssw_displaced_improvedfullseeding_fixInvSIP/roc.json').read()), '10.2.X biased model, correct 1/IP', False),

   ## (json.loads(open('models/2018Nov01/bdt_bo_improvedfullseeding/roc.json').read()), '9.4.X unbiased model on 9.4.X simulation', False),
   ## (json.loads(open('models/2019Feb05/cmssw_unbiased_bdt/roc.json').read()), '9.4.X unbiased model on 10.2.X simulation', False),
   ## (json.loads(open('models/2019Feb05/bdt_cmssw_improvedfullseeding/roc.json').read()), '10.2.X unbiased model, 9.4.X Hyperparameters', False),
   ## (json.loads(open('models/2019Feb05/bdt_bo_cmssw_improvedfullseeding/roc.json').read()), '10.2.X unbiased model, new Bayesian Optimisation', False),

   ## (json.loads(open('models/2018Nov01/bdt_bo_improvedfullseeding/roc.json').read()), '9.4.X unbiased model on 9.4.X simulation', False),
   ## (json.loads(open('models/2019Jan26Selected/94XTraining_improvedfullseeding/roc.json').read()), '9.4.X unbiased model on 10.2.X simulation', False),
   ## (json.loads(open('models/2019Jan26Selected/bdt_improvedfullseeding/roc.json').read()), '10.2.X unbiased model on 10.2.X simulation', False),
   ## (json.loads(open('models/2018Nov01/bdt_bo_improvedfullseeding/roc0p5GeVCut.json').read()), 'seeding + ECAL + HCAL + simple GSF + > 0.5 GeV', True),
   ## (json.loads(open('models/2018Nov01/bdt_bo_displaced_improvedseeding_noweight/roc.json').read()), 'seeding + ECAL + HCAL + displacement, pt biased', True),
   ## (json.loads(open('models/2018Nov01/bdt_bo_displaced_improvedfullseeding_noweight/roc.json').read()), '9.4.X biased model on 9.4.X simulation', False),
   ## (json.loads(open('models/2019Jan26Selected/94XTraining_displaced_improvedfullseeding/roc.json').read()), '9.4.X biased model on 10.2.X simulation', False),
   ## (json.loads(open('models/2019Jan26Selected/bdt_displaced_improvedfullseeding/roc.json').read()), '10.2.X biased model on 10.2.X simulation', False),
   ## (json.loads(open('models/2018Nov01/bdt_bo_displaced_improvedfullseeding_noweight/roc0p5GeVCut.json').read()), 'seeding + ECAL + HCAL + displacement + simple GSF, pt biased + > 0.5 GeV', True),

   ## (json.loads(open('models/2018Oct22v2/bdt_bo_seeding/roc.json').read()), 'seeding 2018Oct22v2'),
   ## (json.loads(open('models/2018Oct22v2/bdt_bo_seeding_2018Oct05Model/roc.json').read()), 'seeding 2018Oct05 on 2018Oct22v2'),
   ## (json.loads(open('models/2018Oct22v2/bdt_seeding_all_Oct05Cfg/roc.json').read()), 'seeding 2018Oct05 Pars'),
   ## (json.loads(open('models/2018Oct22v2/bdt_seeding_BToKee_Oct05Cfg/roc.json').read()), 'seeding 2018Oct05 Pars BToKee'),

   ##(json.loads(open('models/2018Oct05/bdt_bo_seeding/roc.json').read()), 'seeding 2018Oct05'),
   ##(json.loads(open('models/2018Oct05/bdt_bo_seeding_2018Oct22v2Model/roc.json').read()), 'seeding 2018Oct22v2 on 2018Oct05'),
   ##(json.load(open('models/2018Oct05/bdt_seeding_BToKee_Oct05Cfg_2018Oct22v2Data/roc.json')), 'seeding 2018Oct05 Pars'),
   ##(json.load(open('models/2018Oct05/bdt_seeding_all_Oct05Cfg_2018Oct22v2Data/roc.json')), 'seeding 2018Oct05 Pars BToKee'),

   ## (json.loads(open('models/2018Oct22v2/bdt_bo_seeding/roc.json').read()), 'seeding'),
   ## (json.loads(open('models/2018Oct22v2/bdt_bo_fullseeding/roc.json').read()), 'seeding + simple GSF'),
   ## (json.loads(open('models/2018Oct22v2/bdt_bo_improvedseeding/roc.json').read()), 'seeding + cluster shapes'),
   ## (json.loads(open('models/2018Oct22v2/bdt_bo_displaced_seeding/roc.json').read()), 'seeding + displacement'),

   ## (json.loads(open('franken_plots/mod_2018Oct22v2_on_2018Oct05/roc.json').read()), 'Train Oct22v2 test Oct05'),
   ## (json.loads(open('franken_plots/mod_2018Oct22v2_on_2018Oct05_ptcut/roc.json').read()), 'Train Oct22v2 test Oct05 pt cut'),
   ## (json.loads(open('franken_plots/mod_2018Oct22v2_on_2018Oct22v2/roc.json').read()), 'Train Oct22v2 test Oct22v2'),
   ## (json.loads(open('franken_plots/mod_2018Oct22v2_BToKee_on_2018Oct05/roc.json').read()), 'Train Oct22v2 BToKee test Oct05'),
   ## (json.loads(open('franken_plots/mod_2018Oct22v2_BToKee_on_2018Oct05_ptcut/roc.json').read()), 'Train Oct22v2 BToKee test Oct05 pt cut'),
   ## (json.loads(open('franken_plots/mod_2018Oct22v2_BToKee_on_2018Oct22v2/roc.json').read()), 'Train Oct22v2 BToKee test Oct22v2'),
   ## (json.loads(open('franken_plots/mod_2018Oct05_on_2018Oct05/roc.json').read()), 'Train Oct05 test Oct05'),
   ## (json.loads(open('franken_plots/mod_2018Oct05_on_2018Oct05_ptcut/roc.json').read()), 'Train Oct05 test Oct05 pt cut'),
   ## (json.loads(open('franken_plots/mod_2018Oct05_on_2018Oct22v2/roc.json').read()), 'Train Oct05 test Oct22v2'),
   ## (json.loads(open('franken_plots/mod_2018Oct05_ptcut_on_2018Oct05/roc.json').read()), 'Train Oct05 pt cut test Oct05'),
   ## (json.loads(open('franken_plots/mod_2018Oct05_ptcut_on_2018Oct05_ptcut/roc.json').read()), 'Train Oct05 pt cut test Oct05 pt cut'),
   ## (json.loads(open('franken_plots/mod_2018Oct05_ptcut_on_2018Oct22v2/roc.json').read()), 'Train Oct05 pt cut test Oct22v2'),
]

plt.figure(figsize=[8, 8])
plt.plot(
   np.arange(0,1,0.01),
   np.arange(0,1,0.01),
   'k--')

xfrs = [seedings[0][0]['baseline_mistag'], seedings[0][0]['baseline_ptcut_mistag'], seedings[0][0]['baseline_ptcut_mistag']*3, 10*seedings[0][0]['baseline_ptcut_mistag']]
xticks_to_add = [i for i in xfrs]
yticks_to_add = [seedings[0][0]['baseline_eff'], seedings[0][0]['baseline_ptcut_eff']]
## plt.plot([seedings[0][0]['baseline_mistag']], [seedings[0][0]['baseline_eff']], 
##          'o', markersize=5, c='b')

plt.plot([seedings[0][0]['baseline_ptcut_mistag']], [seedings[0][0]['baseline_ptcut_eff']], 
         'o', markersize=5, c='b', label = '10.2.X truth baseline')

## plt.plot([seedings[1][0]['baseline_ptcut_mistag']], [seedings[1][0]['baseline_ptcut_eff']], 
##          'o', markersize=5, c='g', label = '9.4.X truth baseline')

if os.path.isfile('model_or.json'):
   or_model = json.load(open('model_or.json'))
   plt.plot(
      [or_model[i][0] for i in 'LMT'],
      [or_model[i][1] for i in 'LMT'],
      '^', markersize=5, c='g', label='OR model')


for seeding, label, ticks in seedings:
   plt.plot(*seeding['roc'], label=label)
   if ticks:
      for fr in xfrs:
         idx = np.abs(np.array(seeding['roc'][0]) - fr).argmin()
         yticks_to_add.append(seeding['roc'][1][idx])

def merge_ticks(t1, t2, min_delta=0.02, log=False):
   ret = []
   for i in t1+t2:
      if not ret: ret.append(i)
      if log:
         min_d = min(abs(np.log(j/i)) for j in ret)
      else:
         min_d = min(abs(i-j) for j in ret)
      if min_d > min_delta: ret.append(i)
   return ret

from pdb import set_trace
plt.xlabel('Mistag Rate')
plt.ylabel('Efficiency')
ensure = plt.legend(bbox_to_anchor=(1e-4, 1.4), loc=2, borderaxespad=0.)
plt.xlim(0., 1)
plt.grid()
plt.xticks(
   merge_ticks(
      sorted(xticks_to_add),
      list(plt.xticks()[0])
      )
   )
plt.yticks(
   merge_ticks(
      sorted(yticks_to_add),
      list(plt.yticks()[0])
      )
)
plt.savefig('%s/BDT_comparison.png' % plots, 
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.savefig('%s/BDT_comparison.pdf' % plots, 
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.gca().set_xscale('log')
plt.xticks(
   merge_ticks(
      sorted(xticks_to_add),
      list(plt.xticks()[0]),
      log=True
      )
   )
plt.yticks(
   merge_ticks(
      sorted(yticks_to_add),
      list(plt.yticks()[0])
      )
)
plt.xlim(1e-4, 1)
plt.savefig('%s/log_BDT_comparison.png' % plots,
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.savefig('%s/log_BDT_comparison.pdf' % plots,
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.clf()


#
# Plot IDs
#
for seeding, label, ticks in ids:
   plt.plot(*seeding['roc'], label=label)

plt.xlabel('Mistag Rate')
plt.ylabel('Efficiency')
ensure = plt.legend(bbox_to_anchor=(1e-4, 1.4), loc=2, borderaxespad=0.)
plt.xlim(0., 1)
plt.grid()
plt.savefig('%s/ID_comparison.png' % plots, 
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.savefig('%s/ID_comparison.pdf' % plots, 
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.gca().set_xscale('log')
plt.xlim(1e-4, 1)
plt.savefig('%s/log_ID_comparison.png' % plots,
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.savefig('%s/log_ID_comparison.pdf' % plots,
            bbox_extra_artists=(ensure,), bbox_inches='tight')


## plt.figure(figsize=[8, 8])
## plt.title('ID training')
## plt.plot(
##    np.arange(0,1,0.01),
##    np.arange(0,1,0.01),
##    'k--')
## 
## plt.plot(*combined['roc'], label='optimized')
## plt.plot(*old['validation'], label='default')
## 
## plt.xlabel('Mistag Rate')
## plt.ylabel('Efficiency')
## plt.legend(loc='best')
## plt.xlim(0., 1)
## plt.savefig('optimization_comparison.png')
## plt.savefig('optimization_comparison.pdf')
## plt.gca().set_xscale('log')
## plt.xlim(1e-4, 1)
## plt.savefig('log_optimization_comparison.png')
## plt.savefig('log_optimization_comparison.pdf')
## plt.clf()

#ROCs by pT
import matplotlib.lines as mlines
from itertools import cycle
lstyles = cycle(['-', '--', ':', '-.'])
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
pt_ranges = [i for i in seedings[0][0].keys() if i.startswith('trk_pt#')]

## plt.figure(figsize=[8, 8])
## plt.title('ID training')
## plt.plot(
##    np.arange(0,1,0.01),
##    np.arange(0,1,0.01),
##    'k--')
## 
## entries = []
## from pdb import set_trace
## for rr, col in zip(pt_ranges, colors):
##    prange = tuple(rr.split('#')[1].split('to'))
##    plt.plot(*new_id[rr]['roc'], c=col, ls='-')
##    plt.plot(*mva_like[rr]['roc'], c=col, ls=':')
##    plt.plot(*combined[rr]['roc'], c=col, ls='--')
##    entries.append(mlines.Line2D([], [], color=col, label='pT [%s, %s)' % prange))
##    #plt.plot(*seeding[rr]['roc'], c=col, ls='-')
##    #plt.plot(*full_seeding[rr]['roc'], c=col, ls='-')
## 
## entries.extend([
##       mlines.Line2D([], [], color='k', ls='-', label='ID'),
##       mlines.Line2D([], [], color='k', ls=':', label='MVA ID retrain'),
##       mlines.Line2D([], [], color='k', ls='--', label='Combined ID'),
##       ])
##    
## plt.xlabel('Mistag Rate')
## plt.ylabel('Efficiency')
## plt.legend(handles=entries, loc='best')
## plt.xlim(0., 1)
## plt.savefig('BDT_comparison_bypt.png')
## plt.savefig('BDT_comparison_bypt.pdf')
## plt.gca().set_xscale('log')
## plt.xlim(1e-4, 1)
## plt.savefig('log_BDT_comparison_bypt.png')
## plt.savefig('log_BDT_comparison_bypt.pdf')
## plt.clf()


plt.figure(figsize=[8, 8])
plt.plot(
   np.arange(0,1,0.01),
   np.arange(0,1,0.01),
   'k--')

entries = []
for rr, col in zip(pt_ranges, colors):
   lstyles = cycle(['-', '--', ':', '-.'])
   prange = tuple(rr.split('#')[1].split('to'))
   entries.append(mlines.Line2D([], [], color=col, label='pT [%s, %s)' % prange))
   for info, ls in zip(seedings, lstyles):
      seeding, label, _ = info
      plt.plot(*seeding[rr]['roc'], c=col, ls=ls)
   
   plt.plot([seedings[0][0][rr]['baseline_mistag']], [seedings[0][0][rr]['baseline_eff']], 
            'o', markersize=5, c=col)

lstyles = cycle(['-', '--', ':', '-.'])
for info, ls in zip(seedings, lstyles):
   _, label, _ = info
   entries.append(
      mlines.Line2D([], [], color='k', ls=ls, label=label)
      )

entries.extend([
      mlines.Line2D([], [], marker='o', color='k', ls='', label='baseline'),
      ])
   
plt.xlabel('Mistag Rate')
plt.ylabel('Efficiency')
ensure = plt.legend(
   bbox_to_anchor=(1e-4, 1.55), loc=2, borderaxespad=0.,
   handles=entries,
)
#plt.tight_layout()
plt.xlim(0., 1)
plt.savefig('%s/BDT_seeding_comparison_bypt.png' % plots,
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.savefig('%s/BDT_seeding_comparison_bypt.pdf' % plots,
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.gca().set_xscale('log')
plt.xlim(1e-4, 1)
plt.savefig('%s/log_BDT_seeding_comparison_bypt.png' % plots,
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.savefig('%s/log_BDT_seeding_comparison_bypt.pdf' % plots,
            bbox_extra_artists=(ensure,), bbox_inches='tight')
plt.clf()


## combined = [i.split()[0] for i in open('models/2018Oct22v2/bdt_bo_combined_id/importances.txt')]
## mva = [i.split()[0] for i in open('models/2018Oct22v2/bdt_bo_mva_id/importances.txt')]
## naive = [i.split()[0] for i in open('models/2018Oct22v2/bdt_bo_id/importances.txt')]
## 
## max_size = max(len(i) for i in combined)
## form = '%'+str(max_size)+'s   %3s  %3s  %3s\n'
## outf = open('importances.txt', 'w')
## outf.write(form % (' ', 'cmb', 'mva', 'id'))
## for idx, name in enumerate(combined):
##    idx_mva = mva.index(name) if name in mva else '--'
##    idx_naive = naive.index(name) if name in naive else '--'
##    outf.write(form % (name, idx, idx_mva, idx_naive))
## 
## combined_importances = [(i.split()[0], float(i.split()[1])) for i in open('models/2018Oct22v2/bdt_bo_combined_id/importances.txt')]
## tot = 0
## print 'This features could be safely dropped'
## for n, v in combined_importances[::-1]:
##   tot += v
##   if tot > 0.05: break
##   print n

