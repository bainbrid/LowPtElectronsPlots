import numpy as np
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
   '--allTracks', action='store_true', help='use all tracks'
)
parser.add_argument(
   '--fakes', action='store_true', help='use all tracks'
)
parser.add_argument(
   '--test', help='pass a test file'
)
args = parser.parse_args()


import uproot
import matplotlib.pyplot as plt
import root_numpy
import rootpy
import rootpy.plotting as rplt
import json
import pandas as pd
from matplotlib import rc
from pdb import set_trace
import os
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from baseline import baseline
from glob import glob

debug = False
print 'Getting the data'
from datasets import get_data_sync, input_files, tag
plot_type = 'efficiency' if not args.fakes else 'fakerate'

def plot_efficiency(eff, **kw):
   graph = eff.graph   
   effs = [i for i in graph.y()]
   errs = np.array([i for i in graph.yerr()]).transpose()
   xs = [i for i in graph.x()]
   if 'offset' in kw:
      xs = [i+kw['offset'] for i in xs]
      del kw['offset']
   xerr = np.array([i for i in graph.xerr()]).transpose()
   plt.errorbar(xs, effs, yerr=errs, xerr=xerr, **kw)

class EfficiencyEncoder(json.JSONEncoder):
   def default(self, obj):
      if isinstance(obj, rplt.Efficiency):
         graph = eff.graph
         xs = [i for i in graph.x()]
         xerr = [i for i in graph.xerr()]
         effs = [i for i in graph.y()]
         ret = [[i-j[0], i+j[1], k] for i, j, k in zip(xs, xerr, effs)]
         return ret.__repr__()
      return super(EfficiencyEncoder, self).default(obj)

jinfo = {}
for dataset in ['BToKeeByDR', 'BToKeeByHits'] if not args.test else ['current_test']:
   jmap_efficiencies = {}
   if args.test: 
      input_files['current_test'] = glob(args.test)
   print 'plotting for', dataset
   mc = pd.DataFrame(
      get_data_sync(
         dataset, 'all', 
         exclude={'gsf_hit_dpt', 'gsf_hit_dpt_unc', 'gsf_ecal_cluster_ematrix', 'ktf_ecal_cluster_ematrix'}
         )
      )
   mc['baseline'] = (
      mc.preid_trk_ecal_match | 
      (np.invert(mc.preid_trk_ecal_match) & mc.preid_trkfilter_pass & mc.preid_mva_pass)
      )
   
   electrons = mc[mc.is_e == 1 & (np.abs(mc.gen_eta) < 2.4)] if not args.allTracks else mc[(np.abs(mc.trk_eta) < 2.4)]
   if args.fakes:
      electrons = mc[mc.is_other == 1 & (np.abs(mc.trk_eta) < 2.4) & (mc.trk_pt > 0.)]
   histos = {}
   seedings = [
      ('standard', (electrons.baseline & (np.abs(electrons.trk_eta) < 2.4) & (electrons.trk_pt > 2))),
      ('relaxed',  (electrons.baseline & (np.abs(electrons.trk_eta) < 2.4))),
      ##('ECALmatch', ((electrons.trk_pt > 0) & (np.abs(electrons.trk_eta) < 2.4) & (electrons.preid_trk_ecal_Deta < 999))),
      ##('NOTECALmatch', ((electrons.trk_pt > 0) & (np.abs(electrons.trk_eta) < 2.4) & (electrons.preid_trk_ecal_Deta > 999))),
      ##('removed', ((electrons.trk_pt > 0) & (np.abs(electrons.trk_eta) < 2.4)))
      ]

   all_efficiencies = {}
   for seed_name, seeding in seedings:      
      has_gsf = seeding & (electrons.gsf_pt > 0)
      ordered_masks = [
         ('all', None), 
         ('KTF Track', ((electrons.trk_pt > 0) & (np.abs(electrons.trk_eta) < 2.4))),
         ('seeding', seeding),
         ('GSF Track', seeding & (electrons.gsf_pt > 0)),

         ('PFGSF has ktf', has_gsf & electrons.pfgsf_gsf_has_ktf),
         ('PFGSF Fifth step trk', has_gsf & electrons.pfgsf_ktf_is_fifthStep),
         ('PFGSF ECAL Driven', has_gsf & electrons.pfgsf_gsf_ecalDriven),
         ('PFGSF Tracker Driven', has_gsf & electrons.pfgsf_gsf_trackerDriven),
         ('PFGSF valid brem', has_gsf & electrons.pfgsf_valid_gsf_brem),
         ('PFGSF preselection', has_gsf & electrons.pfgsf_passes_preselection),
         ('PFGSF arbitration', has_gsf & electrons.pfgsf_passes_selection),

         ('PF GSFTrk', seeding & (electrons.gsf_pt > 0) & electrons.has_pfGSF_trk),
         ('PF Block', seeding & (electrons.gsf_pt > 0) & electrons.has_pfBlock),
         ('PF Block+ECAL', seeding & (electrons.gsf_pt > 0) & electrons.has_pfBlock_with_ECAL),
         ('PF Ele', seeding & (electrons.gsf_pt > 0) & electrons.has_pfEgamma),
         ('GED Core', seeding & (electrons.gsf_pt > 0) & electrons.has_ele_core),
         ('GED Electrons', seeding & (electrons.ele_pt > 0)),
         ]
      to_plot = {'KTF Track', 'seeding', 'GSF Track', 'GED Electrons'}
      masks = dict(ordered_masks)
      for name, mask in masks.iteritems():
         hist = rplt.Hist([1,2,4,5,6,7,8,9,10] if not args.test else [0,1,2,5,10])
         masked = electrons[mask] if mask is not None else electrons
         root_numpy.fill_hist(hist, masked.gen_pt if not (args.allTracks or args.fakes) else masked.trk_pt)
         histos[name] = hist
         
      efficiencies = {}
      markersize = 6
      first = True
      plt.clf()
      offset = 0.1*(len(masks)-1)/2
      for passing, _ in ordered_masks:
         if passing == 'all': continue
         efficiencies[passing] = rplt.Efficiency(histos[passing], histos['all'])
         if passing not in to_plot: continue
         plot_efficiency(efficiencies[passing], #offset=offset, 
                         fmt="o", markersize=markersize, 
                         label=passing, markeredgewidth=0.0)
         offset -= 0.1
         first = False
      
      all_efficiencies[seed_name] = efficiencies
      plt.legend(loc='lower right')
      plt.title('%s seeding' % seed_name)
      plt.xlabel('gen $p_{T}$' if not (args.allTracks or args.fakes) else 'ktf $p_{T}$')
      plt.ylabel('efficiency')
      plt.ylim(0., 1.)
      plt.grid()
      if args.allTracks: dataset += '_trk'
      plt.savefig('plots/%s/%s_seeding_%s_%s.png' % (tag, dataset, seed_name, plot_type))
      plt.savefig('plots/%s/%s_seeding_%s_%s.pdf' % (tag, dataset, seed_name, plot_type))
      plt.clf()

      normalization = float(electrons.shape[0])
      overall_effs = []
      for name, _ in ordered_masks:
         if name == 'all': continue
         eff, err_d, err_u = efficiencies[name].overall_efficiency()
         overall_effs.append((name, eff, err_d, err_u))
      
      names = [i[0] for i in overall_effs]
      xs = range(len(overall_effs))
      effs = [i for _, i, _, _ in overall_effs]
      errs = np.array([(i, j) for _, _, i, j in overall_effs]).transpose()
      plt.errorbar(xs, effs, yerr=errs, fmt="o", markersize=6, markeredgewidth=0.0)

      ax = plt.gca()
      ax.set_xticks(xs)
      ax.set_xticklabels(names, rotation=90)
      plt.xlim(-1, len(overall_effs))
      plt.ylim(0, 1)
      plt.grid()
      plt.savefig('plots/%s/%s_seeding_%s_cutflow.png' % (tag, dataset, seed_name), bbox_inches='tight')
      plt.savefig('plots/%s/%s_seeding_%s_cutflow.pdf' % (tag, dataset, seed_name), bbox_inches='tight')
      plt.clf()
      

   markersize = 3+2*(len(seedings) - 1)
   for seed_name, _ in seedings:
      eff = all_efficiencies[seed_name]['GED Electrons']
      plot_efficiency(
         eff, fmt="o", markersize=markersize,
         label='%s seeding' % seed_name, markeredgewidth=0.0)
      markersize -= 2
   plt.legend(loc='best')
   plt.xlabel('gen $p_{T}$' if not args.allTracks else 'ktf $p_{T}$')
   plt.ylabel('efficiency')
   plt.ylim(0., 1.)
   plt.grid()
   if args.allTracks: dataset += '_trk'
   plt.savefig('plots/%s/%s_GEDElectrons_%s.png' % (tag, dataset, plot_type))
   plt.savefig('plots/%s/%s_GEDElectrons_%s.pdf' % (tag, dataset, plot_type))
   plt.clf()
   jinfo[dataset] = all_efficiencies

with open('plots/%s/%s_%s.json' % (tag, dataset, plot_type), 'w') as jfile:
   jfile.write(
      json.dumps(
         jinfo,cls=EfficiencyEncoder
         )
      )

