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
matplotlib.use('Agg') # choose backend before doing anything else with pyplot! 
#matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
#rc('text', usetex=True)
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties

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
assert os.getcwd().endswith("icenet/standalone/scripts"), print("You must execute this script from within the 'icenet/standalone/scripts' dir!")

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
   
files = [input_data+'/images.root']

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
   'ele_pt','ele_eta','ele_dr','ele_mva_value','ele_mva_id',
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

for file in files :
   tree = uproot.open(file).get('ntuplizer/tree')

   import collections
   events = tree.arrays(['is_e','image_*'], outputtype=collections.namedtuple)

   print('file:',file,'tree:',tree,'type:',type(tree))
   print(tree.show())
   
   nevents = 10
   nevents = min(nevents,len(events.is_e)) if nevents > 0 else len(events.is_e)
   for event in range(nevents) :
      print('event:',event,'type:',type(events),'len:',len(events))
      #example.features.feature['label'].int64_list.value.append(events.is_e[event])
      #example.features.feature['gsf_n'].int64_list.value.append(events.image_gsf_n[event])

   branches = [branch for branch in tree.keys() if branch.startswith(b'image_')]
   for branch in branches : 
      array = tree[branch].array()
      print('branch:',branch,'type:',type(array),'len:',len(array))

   #branch = tree['image_gsf_ref_eta'].array()
   #print('branch:',branch,'type:',type(branch),'len:',) 
   #event = tree[0]
