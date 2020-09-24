# python 2 and 3 compatibility, pip install future and six
from __future__ import print_function
from future.utils import raise_with_traceback
import future
import builtins
import past
import six

from argparse import ArgumentParser
import os
import uproot
import numpy as np
import pandas as pd

# matplotlib imports 
import matplotlib
matplotlib.use('Agg') # choose backend before doing anything else with pyplot! ('macosx'?)
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
#rc('text', usetex=True)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties
import mplhep as hep
#plt.style.use(hep.style.CMS) 

################################################################################
# TO DO ITEMS !!!
# -100. --> nan
# charge*phi

################################################################################
print("##### Command line args #####")

parser = ArgumentParser()
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--nevents',default=1,type=int)
args = parser.parse_args()
print("Command line args:",vars(args))

################################################################################
print("##### Define inputs #####")

print(os.getcwd())
assert os.getcwd().endswith("icenet/standalone/scripts"), \
    "You must execute this script from within the 'icenet/standalone/scripts' dir!"

# I/O directories
input_data='../data'
print("input_data:",input_data)
input_base=os.getcwd()+"/../input"
output_base=os.getcwd()+"/../output"
if not os.path.isdir(input_base) : 
   raise_with_traceback(ValueError('Could not find input_base "{:s}"'.format(input_base)))
print("input_base:",input_base)
if not os.path.isdir(output_base) : 
   os.makedirs(output_base)
print("output_base:",output_base)
   
#files = [input_data+'/images.root']
#files = [input_data+'/temp_miniaod_test.root']
files = [input_data+'/output_numEvent1000_2.root']

################################################################################
print("##### Define features #####")

features = [
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
   'ele_pt','ele_eta','ele_dr',
   'ele_mva_value','ele_mva_value_depth10','ele_mva_value_depth11','ele_mva_value_depth13','ele_mva_value_depth15',
   'ele_mva_id',
   'evt','weight'
]

labelling = [
   'is_e','is_egamma',
   'has_trk','has_seed','has_gsf','has_ele',
   'seed_trk_driven','seed_ecal_driven'
]

columns = features + additional + labelling
columns = list(set(columns))

################################################################################
print("##### Load files #####")

file = files[0]
tree = uproot.open(file).get('ntuplizer/tree')
print("tree.numentries:",tree.numentries)

import collections
#events = tree.arrays(['is_e','image_*'], outputtype=collections.namedtuple)
#events = tree.arrays(['is_e','image_*'])
events = tree.arrays()
print(events.keys())

for key,vals in events.items() :
   print('{} {}'.format(key.decode(),type(vals)))

#vars = [key for key in tree.keys() if key.startswith(b'gsf_') or key.startswith(b'pfgsf_') ]
vars = [b'ele_mva_value',
        #b'ele_mva_value_depth10',
        #b'ele_mva_value_depth11',
        b'ele_mva_value_depth13',
        b'ele_mva_value_depth15',
        b'ele_mva_value_retrained',
        #b'gsf_bdtout1',
        #b'gsf_bdtout2',
        ]
print('\n'.join([var.decode() for var in vars]))

################################################################################
print("##### Some preprocessing #####")

#is_e = events[b'is_e']
#is_egamma = events[b'is_egamma']

#is_gsf = (events[b'gsf_pt'] > 0.)
#is_gsf = is_gsf & (image_gsf_phi_del<1.56) & (abs(events[b'image_gsf_proj_R']-129.)<0.1)

#is_gsf = (events[b'has_gsf']) & (events[b'gsf_pt'] > 0.5) & (np.abs(events[b'gsf_eta']) < 2.5)
#is_pfgsf = (events[b'has_pfgsf']) & (events[b'pfgsf_pt'] > 0.5) & (np.abs(events[b'pfgsf_eta']) < 2.5)

is_e = events[b'is_e']

# IS LOWPT
is_egamma = np.invert(events[b'is_egamma'])
is_gsf = (events[b'has_ele']) & (events[b'ele_pt'] > 0.5) & (np.abs(events[b'ele_eta']) < 2.5)

# IS EGAMMA
#is_egamma = events[b'is_egamma']
#is_gsf = (events[b'has_ele']) & (events[b'ele_pt'] > 0.5) & (np.abs(events[b'ele_eta']) < 2.5)

################################################################################
print("##### Print (pf)gsf_* variables #####")

if True :

   xmin = -20.
   xmax =  20.
   for var in vars :
      print("histogram:",var)
      f, ax = plt.subplots()
      bins = 100
      counts1,bins = np.histogram(events[var][is_e&is_gsf&is_egamma&(events[var]>xmin)].flatten(),bins=bins)
      counts2,bins = np.histogram(events[var][~is_e&is_gsf&is_egamma&(events[var]>xmin)].flatten(),bins=bins)
      ax.hist(x=bins[:-1], 
              bins=bins,
              weights= counts2 / ( np.sum(counts2) if np.sum(counts2) > 0. else 1. ),
                 histtype='step',
              color='red',
              label='bkgd',
              )
      ax.hist(x=bins[:-1], 
              bins=bins,
              weights= counts1 / ( np.sum(counts1) if np.sum(counts1) > 0. else 1. ),
              histtype='step',
              color='green',
              label='signal',
              )
      #hep.histplot(counts,edges)
      plt.xlabel(var)
      plt.ylabel('Counts/bin')
      plt.yscale('log')
      plt.xlim(xmin,xmax)
      plt.ylim(0.00001,2.)
      #hep.cms.text("Internal")
      #hep.mpl_magic()
      plt.legend()
      plt.savefig('../output/plots_vars/{:s}.pdf'.format(var.decode()))
      plt.close()

################################################################################
print("##### Engineered variables #####")

if False :

   # min, max, and diff or eta and phi
   histos = {
      "pt_asymmetry":( events[b'gsf_mode_pt'] - events[b'gsf_pt'] ) / ( events[b'gsf_mode_pt'] + events[b'gsf_pt'] ),
      "pt_ratio":events[b'gsf_mode_pt'] / events[b'gsf_pt'],
      "eta_asymmetry":( events[b'gsf_mode_eta'] - events[b'gsf_eta'] ) / ( events[b'gsf_mode_eta'] + events[b'gsf_eta'] ),
      "eta_ratio":events[b'gsf_mode_eta'] / events[b'gsf_eta'],
      }

   for title,values in histos.items() :
      print("histogram:",title)
      f, ax = plt.subplots()
      counts,bins = np.histogram(values[~is_e&is_egamma&is_gsf].flatten(),bins=100)
      ax.hist(x=bins[:-1], 
              bins=bins,
              weights=counts/np.sum(counts),
              histtype='step',
              color='red',
              label='~is_e&is_egamma&is_gsf'
              )
      counts,bins = np.histogram(values[is_e&is_egamma&is_gsf].flatten(),bins=bins)
      ax.hist(x=bins[:-1], 
              bins=bins,
              weights=counts/np.sum(counts),
              histtype='step',
              color='green',
              label='is_e&is_egamma&is_gsf'
              )
      #hep.histplot(counts,edges)
      plt.xlabel(title)
      plt.xlabel(title)
      plt.ylabel('Counts/bin')
      plt.yscale('log')
      plt.ylim(0.00001,2.)
      #hep.cms.text("Internal")
      #hep.mpl_magic()
      plt.legend()
      plt.savefig('../output/plots_vars/{:s}.pdf'.format(title))
      plt.close()

################################################################################
print("##### 2D histograms #####")

if False :

   def histo_2d(x,y,cut,xlabel,ylabel,title,xlim=(None,None),ylim=(None,None)) :
      #print("histogram:",x,"VS",y,"CUT",cut)
      f, ax = plt.subplots()
      ax.scatter(x[~is_e&cut],y[~is_e&cut],label='~is_e',color='red')
      ax.scatter(x[is_e&cut],y[is_e&cut],label='is_e',color='green')
      plt.xlim(xlim)
      plt.ylim(ylim)
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.title(title)
      plt.legend()
      #hep.cms.text("Internal")
      #hep.mpl_magic()
      plt.savefig('../output/plots_vars/2d/{:s}_VS_{:s}_CUT_{:s}.pdf'.format(xlabel,ylabel,title))
      plt.close()

   histos2d = {
      ( "gsf_pt",
        "gsf_mode_pt", 
        "is_gsf&(events['gsf_pt']<2.0)" ) :
         ( events[b'gsf_pt'],
           events[b'gsf_mode_pt'],
           is_e&is_egamma&is_gsf&(events[b'gsf_pt']<2.0) ),
      ( "gsf_pt",
        "gsf_pt_ov_gsf_mode_pt", 
        "is_gsf&(events['gsf_pt']<2.0)" ) :
         ( events[b'gsf_pt'],
           events[b'gsf_pt']/events[b'gsf_mode_pt'],
           is_e&is_egamma&is_gsf&(events[b'gsf_pt']<2.0) ),
      ( "gsf_eta",
        "gsf_mode_eta", 
        "is_gsf&(events['gsf_pt']<2.0)" ) :
         ( events[b'gsf_eta'],
           events[b'gsf_mode_eta'],
           is_e&is_egamma&is_gsf&(events[b'gsf_pt']<2.0) ),
      ( "gsf_eta",
        "gsf_eta_ov_gsf_mode_eta", 
        "is_gsf&(events['gsf_pt']<2.0)" ) :
         ( events[b'gsf_eta'],
           events[b'gsf_eta']/events[b'gsf_mode_eta'],
           is_e&is_egamma&is_gsf&(events[b'gsf_pt']<2.0) ),
      }
   
   for (xlabel,ylabel,title),(x,y,cut) in histos2d.items() :
      print("histogram:",title)
      histo_2d(x,y,cut,xlabel,ylabel,title)
      
