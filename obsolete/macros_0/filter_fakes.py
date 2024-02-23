import numpy as np
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
import os
from matplotlib.colors import LogNorm

parser = ArgumentParser()
parser.add_argument(
   '--test', action='store_true',
)
args = parser.parse_args()
dataset = 'test' if args.test else 'all'

import matplotlib.pyplot as plt
import uproot
import json
import pandas as pd
from matplotlib import rc
from pdb import set_trace
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from datasets import get_data, tag, apply_weight, get_data_sync
import os

mods = '%s/src/LowPtElectrons/LowPtElectrons/macros/models/%s' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(mods):
   os.makedirs(mods)

plots = '%s/src/LowPtElectrons/LowPtElectrons/macros/plots/%s' % (os.environ['CMSSW_BASE'], tag)
if not os.path.isdir(plots):
   os.makedirs(plots)

print 'Getting dataset "{:s}"...'.format(dataset)
data = pd.DataFrame(
   get_data_sync(dataset, ['trk_pt', 'trk_eta', 'is_e', 'is_e_not_matched', 'is_other'])
)
print '...Done'

# manipulate dataframe 
data = data[np.invert(data.is_e_not_matched)]
data = data[(data.trk_pt > 0) & (np.abs(data.trk_eta) < 2.4) & (data.trk_pt < 15)]
data['log_trkpt'] = np.log10(data.trk_pt)

# (logpt,eta) range to consider
x_bins = 40
x_min = -2.
x_max = 2.
x_range = np.linspace(x_min, x_max, x_bins, endpoint=False)
y_bins = 12
y_min = -3.
y_max = 3.
#y_range = np.linspace(y_min, y_max, y_bins, endpoint=False)
y_range = np.array([-3.,-2.5,-2.,-1.56,-1.44,-0.8,0.,0.8,1.44,1.56,2.,2.5,])
xx, yy = np.meshgrid(x_range, y_range)

#print x_min, y_min, x_max, y_max
#print x_range
#print y_range
#print xx
#print yy

# add (logpt,eta) indices
find_index = np.vectorize(lambda x,y: min(np.searchsorted(x,y)-1,len(x)-1), excluded={0})
data['log_trkpt_index'] = find_index(x_range,data.log_trkpt)
data['trk_eta_index'] = find_index(y_range,data.trk_eta)

# fractions of signal and bkgd in each (ipt,ieta) bin
fractions = []
for (ipt,ieta), group in data.groupby(['log_trkpt_index','trk_eta_index']):
   #print ipt,ieta,float(group.is_e.sum())
   fractions.append( (ipt,
                      ieta,
                      float(group.is_e.sum()) /
                      float(data.is_e.sum()),
                      float(np.invert(group.is_e).sum()) /
                      float(np.invert(data.is_e).sum())) )
print "Number of eles: ",data.is_e.sum()
print "Number of fakes:",np.invert(data.is_e).sum()
   
# Reshape diff signal and bkgd counts for plot, and calc ratio
S_diff = np.zeros(xx.shape)
B_diff = np.zeros(xx.shape)
R_diff = np.zeros(xx.shape)
for (ipt,ieta,sig,bkgd) in fractions : 
   S_diff[ieta,ipt] = sig
   B_diff[ieta,ipt] = bkgd
   R_diff[ieta,ipt] = sig/bkgd if bkgd > 0. else 0.

# Norm max ratio to unity, then set all zeroes to ones
R_norm = R_diff / R_diff.max()
#zeroes = R_norm == 0.
#R_norm[zeroes] = 1.

# Put the result into a color plot
import cosmetics
from matplotlib.colors import LogNorm
plt.figure(figsize=[10, 8])
for vals,name in [ (S_diff,"signal_diff"),
                   (B_diff,"bkgd_diff"),
                   (R_diff,"ratio_diff"),
                   (R_norm,"weights_diff"),
                   ] : 
   plt.imshow(
      vals, interpolation='nearest',
      #extent=(xx.min(), xx.max(), yy.min(), yy.max()),
      extent=(x_min, x_max, y_min, y_max),
      cmap=plt.cm.inferno,
      norm=LogNorm( vmin=0.3*10.**np.log10(np.min(vals[np.nonzero(vals)])),
                    vmax=3.0*10.**np.log10(vals.max()) ),
      aspect='auto', origin='lower')
   plt.title('weight')
   plt.xlim(x_min, x_max)
   plt.ylim(y_min, y_max)
   plt.xlabel(cosmetics.beauty['log_trkpt'])
   plt.ylabel(cosmetics.beauty['trk_ieta'])
   plt.colorbar()
   plt.plot()
   try : 
      print 'Saving %s/%s_%s.png' % (plots, dataset, name)
      plt.savefig('%s/%s_%s.png' % (plots, dataset, name))
   except : pass
   try : 
      print 'Saving %s/%s_%s.pdf' % (plots, dataset, name)
      plt.savefig('%s/%s_%s.pdf' % (plots, dataset, name))
   except : pass
   plt.clf()

# write txt file (contains only nonzero bins)
for vals,name in [ (S_diff,"signal_diff"),
                   (B_diff,"bkgd_diff"),
                   (R_diff,"ratio_diff"),
                   (R_norm,"weights_diff"),
                   ] : 
   filename = '%s/%s_%s.txt' % (mods, dataset, name)
   f = open(filename,"wb")
   output = []
   for (pt,eta,sig) in zip(xx.ravel(),yy.ravel(),vals.ravel()) :
      entry = ( float("{:.2f}".format(pt)),
                float("{:.2f}".format(eta)),
                float("{:f}".format(sig)) )
      # print entry
      output.append(entry)
      f.write("{:.2f}\t{:.2f}\t{:f}\n".format(*entry))
   f.close()
   # write pkl file (contains only nonzero bins)
   name = '%s/%s_%s.pkl' % (mods, dataset, name)
   f = open(name,"wb")
   import pickle
   pickle.dump(output,f)
   f.close()
