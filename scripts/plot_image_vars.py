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
   
files = [
   #input_data+'/images_orig.root'
   #input_data+'/images_phiq.root'
   input_data+'/images_phiq_ref.root'
   ]

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

vars = [key for key in tree.keys() if key.startswith(b'image_')]
print('\n'.join([var.decode() for var in vars]))

################################################################################
print("##### Some preprocessing #####")

# Set all values of -10. to np.nan
#for var in vars : 
#   if 'JaggedArray' in str(type(events[var])) : continue
#   ind = np.isclose(events[var],np.full_like(events[var],-10.))
#   if np.sum(ind) > 0 : events[var][ind] = np.nan

def iphi(phi) : 
   my_func = np.vectorize( lambda phi : int(phi / (2.*3.141592/360.)) )
   return my_func(phi)

#phi_vars = [var for var in vars if 'image' in var.decode() and 'phi' in var.decode() ]
#print(phi_vars)
#for var in phi_vars : 
#   # Multiple phi by charge (='phiq') for values != -10.
#   new_var1 = var.decode().replace('phi','phiq').encode() # just a copy
#   events[new_var1] = events[var]*events[b'image_gsf_charge']
#   mask = ( events[new_var1] > 2.*np.pi )
#   events[new_var1][mask] = -1.*events[new_var1][mask]
##   mask = ( events[var] > -np.pi )
##   events[new_var1] = events[var]
##   events[new_var1][mask] = events[new_var1][mask]*events[b'image_gsf_charge'][mask]
#   # Convert 'phi' to 'iphi'
#   #new_var2 = var.decode().replace('phi','iphi').encode()
#   #events[new_var2] = iphi(events[var])
#   # Convert 'phiq' to 'iphiq'
#   #new_var3 = new_var1.decode().replace('qphi','iphiq').encode()
#   #events[new_var3] = iphi(events[new_var1])
#   print(var,new_var1)
#   print(events[var][:10])
#   print(events[new_var1][:10])

def limit_phi(phi) : # reduce to [-pi,pi]
   my_func = np.vectorize( lambda x : x 
                           if abs(x) <= np.pi 
                           else x - round(x/(2.*np.pi),0)*2.*np.pi )
   return my_func(phi)

def limit_phi2(phi) : # reduce to [0,2pi]
   my_func = np.vectorize( lambda x : x 
                           if x <= 2.*np.pi and x >= 0.
                           else x - round(x/(2.*np.pi),0)*2.*np.pi )
   return my_func(phi)

print("{:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} ".format(          -2.*np.pi,
                                                                   -np.pi,
                                                                   0.,
                                                                   np.pi,
                                                                   2.*np.pi))
print("{:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} ".format(limit_phi(-2.*np.pi), 
                                                        limit_phi(-np.pi), 
                                                        limit_phi(0.), 
                                                        limit_phi(np.pi), 
                                                        limit_phi(2.*np.pi)))
print("{:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} ".format(limit_phi2(-2.*np.pi),
                                                        limit_phi2(-np.pi),limit_phi2(0.),
                                                        limit_phi2(np.pi),
                                                        limit_phi2(2.*np.pi)))

is_e = (events[b'is_e'] == True)
is_gsf = (events[b'image_gsf_inner_pt'] > 0.) & (events[b'image_gsf_atcalo_p'] > 0.)
mask = is_e&is_gsf

inner_eta  = events[b'image_gsf_inner_eta']
proj_eta   = events[b'image_gsf_proj_eta']
atcalo_eta = events[b'image_gsf_atcalo_eta']
atcalo_eta_tmp = np.where(atcalo_eta < -np.pi,           # if value is -10.
                          np.full(atcalo_eta.shape,10.), # replace with +10. (to allow determination of eta_min)
                          atcalo_eta)                    # or keep original

image_gsf_eta_min = np.minimum(inner_eta,proj_eta)
image_gsf_eta_max = np.maximum(inner_eta,proj_eta)
image_gsf_eta_del = image_gsf_eta_max - image_gsf_eta_min

print('eta') 
print([ "{:6.2f} ".format(x) for x in inner_eta[mask][:10]],"inner") 
print([ "{:6.2f} ".format(x) for x in proj_eta[mask][:10]],"proj") 
print([ "{:6.2f} ".format(x) for x in atcalo_eta[mask][:10]],"atcalo") 
print([ "{:6.2f} ".format(x) for x in image_gsf_eta_min[mask][:10]],"min")
print([ "{:6.2f} ".format(x) for x in image_gsf_eta_max[mask][:10]],"max")
print([ "{:6.2f} ".format(x) for x in image_gsf_eta_del[mask][:10]],"max")

inner_phi  = events[b'image_gsf_inner_phi']
proj_phi   = events[b'image_gsf_proj_phi']
atcalo_phi = events[b'image_gsf_atcalo_phi']
atcalo_phi_tmp = np.where(atcalo_phi < -np.pi,           # if value is -10.
                          np.full(atcalo_phi.shape,10.), # replace with +10. (to allow determination of phi_min)
                          atcalo_phi)                    # or keep original

image_gsf_phi_min = proj_phi
image_gsf_phi_max = inner_phi
image_gsf_phi_del = limit_phi(image_gsf_phi_max - image_gsf_phi_min)

is_gsf = is_gsf & (abs(image_gsf_phi_del)<1.56) & (abs(events[b'image_gsf_proj_R']-129.)<0.1)
#print("image_gsf_proj_R",list(zip(*np.histogram(sorted(events[b'image_gsf_proj_R'][mask].flatten()),bins=100))))

print('phi') 
print([ "{:6.2f} ".format(x) for x in inner_phi[mask][:10]],"inner") 
print([ "{:6.2f} ".format(x) for x in proj_phi[mask][:10]],"proj") 
print([ "{:6.2f} ".format(x) for x in atcalo_phi[mask][:10]],"atcalo") 
print([ "{:6.2f} ".format(x) for x in image_gsf_phi_min[mask][:10]],"min") 
print([ "{:6.2f} ".format(x) for x in image_gsf_phi_max[mask][:10]],"max") 
print([ "{:6.2f} ".format(x) for x in image_gsf_phi_del[mask][:10]],"del") 

################################################################################
print("##### Print image_* variables #####")

if True :

   for var in vars :
      print("histogram:",var)
      f, ax = plt.subplots()
      bins = 100
      counts1,bins = np.histogram(events[var][is_e&is_gsf].flatten(),bins=bins)
      counts2,bins = np.histogram(events[var][~is_e&is_gsf].flatten(),bins=bins)
      ax.hist(x=bins[:-1], 
              bins=bins,
              weights= counts2 / ( np.sum(counts2) if np.sum(counts2) > 0. else 1. ),
                 histtype='step',
              color='red',
              label='~is_e&is_gsf'
              )
      ax.hist(x=bins[:-1], 
              bins=bins,
              weights= counts1 / ( np.sum(counts1) if np.sum(counts1) > 0. else 1. ),
              histtype='step',
              color='green',
              label='is_e&is_gsf'
              )
      #hep.histplot(counts,edges)
      plt.xlabel(var)
      plt.ylabel('Counts/bin')
      plt.yscale('log')
      plt.ylim(0.00001,2.)
      #hep.cms.text("Internal")
      #hep.mpl_magic()
      plt.legend()
      plt.savefig('../output/plots_image/{:s}.pdf'.format(var.decode()))
      plt.close()

################################################################################
print("##### Engineered variables #####")

if True :

   # min, max, and diff or eta and phi
   histos = {
      "image_gsf_eta_max":image_gsf_eta_max,
      "image_gsf_eta_min":image_gsf_eta_min,
      "image_gsf_eta_del":image_gsf_eta_del,
      "image_gsf_phi_max":image_gsf_phi_max,
      "image_gsf_phi_min":image_gsf_phi_min,
      "image_gsf_phi_del":image_gsf_phi_del,
      }

   for title,values in histos.items() :
      print("histogram:",title)
      f, ax = plt.subplots()
      counts,bins = np.histogram(values[~is_e&is_gsf].flatten(),bins=100)
      ax.hist(x=bins[:-1], 
              bins=bins,
              weights=counts/np.sum(counts),
              histtype='step',
              color='red',
              label='~is_e&is_gsf'
              )
      counts,bins = np.histogram(values[is_e&is_gsf].flatten(),bins=bins)
      ax.hist(x=bins[:-1], 
              bins=bins,
              weights=counts/np.sum(counts),
              histtype='step',
              color='green',
              label='is_e&is_gsf'
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
      plt.savefig('../output/plots_image/{:s}.pdf'.format(title))
      plt.close()

################################################################################
print("##### 2D histograms #####")

if True :

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
      plt.savefig('../output/plots_image/2d/{:s}_VS_{:s}_CUT_{:s}.pdf'.format(xlabel,ylabel,title))
      plt.close()

   histos2d = {
      ( "image_gsf_inner_pt",
        "image_gsf_eta_del", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)" ) :
         ( events[b'image_gsf_inner_pt'],
           image_gsf_eta_del,
           is_gsf&(events[b'image_gsf_inner_pt']<2.0) ),
      ( "image_gsf_inner_eta",
        "image_gsf_eta_del", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)" ) :
         ( events[b'image_gsf_inner_eta'],
           image_gsf_eta_del,
           is_gsf&(events[b'image_gsf_inner_pt']<2.0) ),
      ( "image_gsf_proj_eta",
        "image_gsf_eta_del", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)" ) :
         ( events[b'image_gsf_proj_eta'],
           image_gsf_eta_del,
           is_gsf&(events[b'image_gsf_inner_pt']<2.0) ),
      ( "image_gsf_phi_del",
        "image_gsf_eta_del", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)" ) :
         ( image_gsf_phi_del,
           image_gsf_eta_del,
           is_gsf&(events[b'image_gsf_inner_pt']<2.0) ),
      ( "image_gsf_inner_pt",
        "image_gsf_phi_del", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)" ) :
         ( events[b'image_gsf_inner_pt'],
           image_gsf_phi_del,
           is_gsf&(events[b'image_gsf_inner_pt']<2.0) ),
      ( "image_gsf_phi_min",
        "image_gsf_phi_del", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)" ) :
         ( image_gsf_phi_min,
           image_gsf_phi_del,
           is_gsf&(events[b'image_gsf_inner_pt']<2.0) ),
      ( "image_gsf_phi_max",
        "image_gsf_phi_del", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)" ) :
         ( image_gsf_phi_max,
           image_gsf_phi_del,
           is_gsf&(events[b'image_gsf_inner_pt']<2.0) ),
      ( "inner_phi",
        "proj_phi", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)" ) :
         ( inner_phi,
           proj_phi,
           is_gsf&(events[b'image_gsf_inner_pt']<2.0) ),
      ( "atcalo_phi",
        "proj_phi", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)" ) :
         ( atcalo_phi,
           proj_phi,
           is_gsf&(events[b'image_gsf_inner_pt']<2.0) ),
      ( "proj_phi-atcalo_phi",
        "proj_R-atcalo_R", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)" ) :
         ( proj_phi-atcalo_phi,
           events[b'image_gsf_proj_R']-events[b'image_gsf_atcalo_R'],
           is_gsf&(events[b'image_gsf_inner_pt']<2.0) ),
      ( "proj_phi-atcalo_phi",
        "proj_R-atcalo_R", 
        "is_gsf&(events['image_gsf_inner_pt']<2.0)&(abs(events[b'image_gsf_proj_eta'])<0.1)" ) :
         ( proj_phi-atcalo_phi,
           events[b'image_gsf_proj_R']-events[b'image_gsf_atcalo_R'],
           is_gsf&(events[b'image_gsf_inner_pt']<2.0)&(abs(events[b'image_gsf_proj_eta'])<0.1) ),
      }
   
   for (xlabel,ylabel,title),(x,y,cut) in histos2d.items() :
      print("histogram:",title)
      histo_2d(x,y,cut,xlabel,ylabel,title)

################################################################################
print("##### Preprocess #####")

#if True :
