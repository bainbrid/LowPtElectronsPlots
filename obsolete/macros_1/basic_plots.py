import numpy as np
import matplotlib
matplotlib.use('Agg')
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
import cosmetics

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
   '--what', default='basic_plots_default', type=str
)
parser.add_argument(
   '--test', action='store_true'
)
parser.add_argument(
   '--multi_dim', action='store_true'
)
args = parser.parse_args()

debug = False
print 'Getting the data'
from datasets import dataset_names, tag, get_data_sync, kmeans_weighter, training_selection, pre_process_data, target_dataset

dataset = 'test' if args.test else target_dataset

plots = '%s/src/LowPtElectrons/LowPtElectrons/macros/plots/%s/' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(plots):
   os.mkdirs(plots)

mods = '%s/src/LowPtElectrons/LowPtElectrons/macros/models/%s/' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(mods):
   os.makedirs(mods)

##all_data = {}
##for dataset in dataset_names:
##   print 'loading', dataset
##   all_data[dataset] = pd.DataFrame(
##      get_data_sync(
##         dataset, 
##         ['is_e', 'is_e_not_matched', 'is_other',
##          'gen_pt', 'gen_eta', 'trk_pt'
##          ]
##         )
##      )
##
##plt.figure(figsize=[8,8])
##for to_plot, nbins in [
##   ('gen_pt', 30),
##   ('gen_eta', 30),
##   ('trk_pt', 30),]:
##   plt.clf()
##   for dataset, sample in all_data.iteritems():
##      electrons = sample[sample.is_e]
##      plt.hist(
##         electrons[to_plot], bins=nbins, 
##         range=cosmetics.ranges[to_plot],
##         histtype='step', normed=True,
##         label = dataset_names[dataset],
##         )
##   plt.xlabel(cosmetics.beauty[to_plot])
##   plt.ylabel('Fraction')
##   plt.legend(loc='best')
##   plt.plot()
##   try : plt.savefig('%s/electrons_%s.png' % (plots, to_plot))
##   except : pass
##   try : plt.savefig('%s/electrons_%s.pdf' % (plots, to_plot))
##   except : pass
##   plt.clf()
##
##   plt.clf()
##   for dataset, sample in all_data.iteritems():
##      electrons = sample[(sample.is_e) & (sample.trk_pt > 0)]
##      plt.hist(
##         electrons[to_plot], bins=nbins, 
##         range=cosmetics.ranges[to_plot],
##         histtype='step', normed=True,
##         label = dataset_names[dataset],
##         )
##   plt.xlabel(cosmetics.beauty[to_plot])
##   plt.ylabel('Fraction')
##   plt.legend(loc='best')
##   plt.plot()
##   try : plt.savefig('%s/electrons_withTrk_%s.png' % (plots, to_plot))
##   except : pass
##   try : plt.savefig('%s/electrons_withTrk_%s.pdf' % (plots, to_plot))
##   except : pass
##   plt.clf()

from features import *
features, additional = get_features(args.what)

# additional features, used somewhere in logic below
features += ['gsf_pt', 'gsf_eta', 'preid_bdtout1']

multi_dim_branches = []
data = None
multi_dim = None
if args.multi_dim :
   multi_dim_branches = ['gsf_ecal_cluster_ematrix', 'ktf_ecal_cluster_ematrix']
   data, multi_dim = pre_process_data(
      dataset,
      features+labeling+additional+multi_dim_branches,
      for_seeding = ('seeding' in args.what)
      )
else :
   data = pre_process_data(
      dataset,
      features+labeling+additional,
      for_seeding = ('seeding' in args.what)
      )

#@@data['eid_sc_Nclus'] = data['sc_Nclus']
#@@features+= ['eid_sc_Nclus']

print 'making plots in dir: ',plots

for feat in multi_dim_branches:
   vals = {}
   for dataset in [
      {'name' : 'electrons',
       'mask' : data.is_e,
       'weight' : data[data.is_e].weight},
      {'name' : 'tracks',
       'mask' : np.invert(data.is_e),
       'weight' : data[np.invert(data.is_e)].weight},
      ]:
      plt.clf()
      plt.title(feat.replace('_', ' '))
      masked = multi_dim[feat][dataset['mask']]
      sum_val = masked.sum(axis=-1).sum(axis=-1)
      mask = np.invert(sum_val == 0)
      masked = masked[mask]
      sum_val = sum_val[mask]
      masked /= sum_val[:,None,None]
      heatmap = np.average(masked, axis=0, weights=dataset['weight'][mask])
      vals[dataset['name']] = heatmap
      plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
      plt.colorbar()
      try : plt.savefig('%s/%s_%s.png' % (plots, dataset['name'], feat))
      except : pass
      try : plt.savefig('%s/%s_%s.pdf' % (plots, dataset['name'], feat))
      except : pass
      plt.clf()
   #make ratios
   ratio = (vals['electrons']/vals['tracks'])-1
   plt.clf()
   plt.title(feat.replace('_', ' '))
   plt.imshow(ratio, cmap='RdBu', interpolation='nearest', vmin=-1, vmax=1)
   plt.colorbar()
   try : plt.savefig('%s/ratio_%s_%s.png' % (plots, dataset['name'], feat))
   except : pass
   try : plt.savefig('%s/ratio_%s_%s.pdf' % (plots, dataset['name'], feat))
   except : pass
   plt.clf()

mask_gsf   = ( (data.gsf_pt > 0.5) & (np.abs(data.gsf_eta) < 2.5) )
mask_ele   = ( mask_gsf & (data.eid_ele_pt > 0) )
mask_ele_V = ( mask_ele & (data.preid_bdtout1 > 0.19) )
mask_ele_L = ( mask_ele & (data.preid_bdtout1 > 1.20) )
mask_ele_M = ( mask_ele & (data.preid_bdtout1 > 2.02) )
mask_ele_T = ( mask_ele & (data.preid_bdtout1 > 3.05) )
data_gsf   = data[mask_gsf]
data_ele   = data[mask_ele_V]
data_ele_L = data[mask_ele_L]
data_ele_T = data[mask_ele_T]

datas = {
   'full_elecs' : {
      'trk' : data_ele[np.invert(data_ele.is_e)],
      'ele' : data_ele_L[data_ele_L.is_e],
      'ele_T' : data_ele_T[data_ele_T.is_e],
      },
   'data' : {
      'trk' : data[np.invert(data.is_e)],
      'ele' : data[data.is_e],
      }
   }

from collections import OrderedDict as odict
dct = odict([
   ('ele','Electrons'),
   ('trk','Tracks'),
   ('ele_T','Electrons (T WP)'),
   ])

for to_plot in features:
   print ' --> plotting', to_plot
   plt.clf()
   if to_plot.startswith('eid_') :
      for idx,name in dct.items() :
         plt.hist(
            datas['full_elecs'][idx][to_plot], bins=50,
            weights=datas['full_elecs'][idx].weight,
            range=cosmetics.ranges.get(to_plot, None),
            histtype='step', normed=True,
            label = name,
            )
   else :
      plt.hist(
         datas['data']['ele'][to_plot], bins=50,
         weights=datas['data']['ele'].weight,
         range=cosmetics.ranges.get(to_plot, None),
         histtype='step', normed=True,
         label = 'Electrons',
         )
      plt.hist(
         datas['data']['trk'][to_plot], bins=50,
         weights=datas['data']['trk'].weight,
         range=cosmetics.ranges.get(to_plot, None),
         histtype='step', normed=True,
         label = 'Tracks',
         )

   plt.xlabel(cosmetics.beauty.get(to_plot, to_plot.replace('_', ' ')))
   plt.ylabel('Fraction')
   plt.legend(loc='best')
   plt.plot()
   try : plt.savefig('%s/electrons_vs_tracks_%s.png' % (plots, to_plot))
   except : pass
   try : plt.savefig('%s/electrons_vs_tracks_%s.pdf' % (plots, to_plot))
   except : pass
   plt.clf()
