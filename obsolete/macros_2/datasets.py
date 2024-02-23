from glob import glob
#A single place where to bookkeep the dataset file locations
#tag = '2018Sep20'
#tag = '2019Feb05'
#posix = '2019Feb05'

tag = '2019Aug13'
posix = '2019Aug13'
target_dataset = 'test'

import socket
path = ""
if "cern.ch" in socket.gethostname() : path = '/eos/cms/store/cmst3/group/bpark/electron_training'
elif 'cmg-gpu1080' in socket.gethostname() : path = '/eos/cms/store/cmst3/group/bpark/electron_training'
elif "hep.ph.ic.ac.uk" in socket.gethostname() : path = '/vols/cms/bainbrid/BParking/electron_training'
print socket.gethostname()

import os
from pdb import set_trace
all_sets = glob(path+'/*_%s_*.root' % posix)
sets = set([os.path.basename(i).split('_')[0].split('Assoc')[0] for i in all_sets])
sets = sorted(list(sets), key=lambda x: -len(x))
input_files = {i : [] for i in sets}
input_files['all'] = all_sets
for inf in all_sets:
   for name in sets:
      if os.path.basename(inf).startswith(name):
         input_files[name].append(inf)
         break
input_files['test'] = [
   '/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15/src/2-ntuples-from-crab/output.CURRENT.root'
   #'/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15/src/2-ntuples-from-crab/output_numEvent1000.root'
   #'/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15/src/2-ntuples-from-crab/output_test.root'
   #'/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15/src/2-ntuples-from-crab/output_new.root'
   #'/afs/cern.ch/user/b/bainbrid/work/public/7-slc7/CMSSW_10_2_15/src/2-ntuples-from-crab/output_old.root'
   #
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_1.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_2.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_3.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_4.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_5.root',
#   '/eos/cms/store/cmst3/group/bpark/electron_training/2019Jul22/MINIAOD/output_6.root',
]

dataset_names = {
   'BToKee' : r'B $\to$ K ee',
   #'BToKstee' : r'B $\to$ K* ee',
   'BToJPsieeK' : r'B $\to$ K J/$\Psi$(ee)',
   'BToJPsieeK_0' : r'B $\to$ K J/$\Psi$(ee)',
   #'BToKstJPsiee' : r'B $\to$ K* J/$\Psi$(ee)',
}

import os
if not os.path.isdir('plots/%s' % tag):
   os.mkdir('plots/%s' % tag)

#import concurrent.futures
import multiprocessing
import uproot
import numpy as np

def get_models_dir():
   if 'CMSSW_BASE' not in os.environ:
      cmssw_path = dir_path = os.path.dirname(os.path.realpath(__file__)).split('src/LowPtElectrons')[0]
      os.environ['CMSSW_BASE'] = cmssw_path
   
   mods = '%s/src/LowPtElectrons/LowPtElectrons/macros/models/%s/' % (os.environ['CMSSW_BASE'], tag)
   if not os.path.isdir(mods):
      os.makedirs(mods)
   return mods

## def get_data(dataset, columns, nthreads=2*multiprocessing.cpu_count(), exclude={}):
##    thread_pool = concurrent.futures.ThreadPoolExecutor(nthreads)
##    if dataset not in input_files:
##       raise ValueError('The dataset %s does not exist, I have %s' % (dataset, ', '.join(input_files.keys())))
##    infiles = [uproot.open(i) for i in input_files[dataset]]
##    if columns == 'all':
##       columns = [i for i in infiles[0]['features/tree'].keys() if i not in exclude]
##    ret = None
##    arrays = [i['features/tree'].arrays(columns, executor=thread_pool, blocking=False) for i in infiles]
##    ret = arrays[0]()   
##    for arr in arrays[1:]:
##       tmp = arr()
##       for column in columns:
##          ret[column] = np.concatenate((ret[column],tmp[column]))
##    return ret

def get_data_sync(dataset, columns, nthreads=2*multiprocessing.cpu_count(), exclude={}, path='ntuplizer/tree'):
   if dataset not in input_files:
      raise ValueError('The dataset %s does not exist, I have %s' % (dataset, ', '.join(input_files.keys())))
   print 'Getting files from "%s": ' % dataset
   print ' \n'.join(input_files[dataset])
   infiles = [uproot.open(i) for i in input_files[dataset]]
   print 'available branches:\n',infiles[0][path].keys()
   if columns == 'all':
      columns = [i for i in infiles[0][path].keys() if i not in exclude]
   try:
      ret = infiles[0][path].arrays(columns)
   except KeyError as ex:
      print 'Exception! ', ex
      set_trace()
      raise RuntimeError('Failed to open %s properly' % infiles[0])
   for infile in infiles[1:]:
      try:
         arrays = infile[path].arrays(columns)
      except:
         raise RuntimeError('Failed to open %s properly' % infile)         
      for column in columns:
         ret[column] = np.concatenate((ret[column],arrays[column]))
   return ret

from sklearn.cluster import KMeans
from sklearn.externals import joblib
import json
from pdb import set_trace

apply_weight = np.vectorize(lambda x, y: y.get(x), excluded={2})

def kmeans_weighter(features, fname):
   kmeans = joblib.load(fname)
   cluster = kmeans.predict(features)
   str_weights = json.load(open(fname.replace('.pkl', '.json')))
   weights = {}
   for i in str_weights:
      try:
         weights[int(i)] = str_weights[i]
      except:
         pass
   return apply_weight(cluster, weights)

def training_selection(df,low=0.5,high=15.):
   #'ensures there is a GSF Track and a KTF track within eta/pt boundaries'
   return (df.trk_pt > low) & (df.trk_pt < high) & (np.abs(df.trk_eta) < 2.4) #@@ original filter
   #return (df.gen_pt < 0.) | (df.gen_pt > 0.) #@@ use for AxE performance

import rootpy.plotting as rplt
import root_numpy

class HistWeighter(object):
   def __init__(self, fname):
      values = [[float(i) for i in j.split()] for j in open(fname)]
      vals = np.array(values)
      xs = sorted(list(set(vals[:,0])))
      ys = sorted(list(set(vals[:,1])))
      vals[:,0] += 0.0001
      vals[:,1] += 0.0001
      mask = (vals[:,2] == 0)
      vals[:,2][mask] = 1 #turn zeros into ones
      vals[:,2] = 1/vals[:,2]
      self._hist = rplt.Hist2D(xs, ys)
      root_numpy.fill_hist(self._hist, vals[:,[0,1]], vals[:, 2])

   def _get_weight(self, x, y):
      ix = self._hist.xaxis.FindFixBin(x)
      iy = self._hist.yaxis.FindFixBin(y)
      return self._hist.GetBinContent(ix, iy)
   def get_weight(self, x, y):
      cnt = lambda x, y: self._get_weight(x, y)
      cnt = np.vectorize(cnt)
      return cnt(x, y)

import pandas as pd
import numpy as np
def pre_process_data(dataset, features, for_seeding=False, keep_nonmatch=False):  
   mods = get_models_dir()
   #features = list(set(features+['trk_pt', 'gsf_pt', 'trk_eta', 'gsf_charge', 'evt', 'gsf_eta']))
   features = list(set(features+['gen_pt', 'gen_eta', 
                                 'trk_pt', 'trk_eta', 'trk_charge', 'trk_dr',
                                 'seed_trk_driven', 'seed_ecal_driven',
                                 'gsf_pt', 'gsf_eta', 'gsf_dr', 'ele_dr',
                                 'pfgsf_pt', 'pfgsf_eta', #@@
                                 'ele_pt', 'ele_eta', 'ele_dr',
                                 'evt', 'weight']))
   data_dict = get_data_sync(dataset, features) # path='features/tree')


   print "Pre-processing data ..."
   if 'is_e_not_matched' not in data_dict:
      data_dict['is_e_not_matched'] = np.zeros(data_dict['trk_pt'].shape, dtype=bool)
   multi_dim = {}
   for feat in ['gsf_ecal_cluster_ematrix', 'ktf_ecal_cluster_ematrix']:
      if feat in features:
         multi_dim[feat] = data_dict.pop(feat, None)
   data = pd.DataFrame(data_dict)
   #data = data.head(10000) #@@ useful for testing

   ##FIXME
   ##if 'gsf_ecal_cluster_ematrix' in features:
   ##   flattened = pd.DataFrame(multi_dim['gsf_ecal_cluster_ematrix'].reshape(multi_dim.shape[0], -1))
   ##   new_features = ['crystal_%d' % i for i in range(len(flattened.columns))]
   ##   flattened.columns = new_features
   ##   features += new_features
   ##   data = pd.concat([data, flattened], axis=1)

   # hack to rename weight column to prescale column
   data['prescale'] = data.weight
   data['weight'] = np.ones(data.weight.shape)

   #remove non-matched electrons
   if not keep_nonmatch:
      multi_dim = {i : j[np.invert(data.is_e_not_matched)] for i, j in multi_dim.iteritems()}
      data = data[np.invert(data.is_e_not_matched)] 
   else:
      #make the right fraction as is_e_not_matched are fully kept and normal tracks have a 160 prescale
      notmatched = data[data.is_e_not_matched]
      data = data[np.invert(data.is_e_not_matched)]
      mask = np.random.uniform(size=notmatched.shape[0]) < 1./160
      notmatched = notmatched[mask]
      data = pd.concat((data, notmatched))
   # training pre-selection
   #mask = training_selection(data) #@@ if used here, cannot determine AxE performance
   #multi_dim = {i : j[mask] for i, j in multi_dim.iteritems()}   
   #data = data[mask]
   if 'trk_dxy' in data_dict and 'trk_dxy_err' in data_dict:
      sip = data.trk_dxy/data.trk_dxy_err
      sip[np.isinf(sip)] = 0
      data['trk_dxy_sig'] = sip
      inv_sip = data.trk_dxy_err/data.trk_dxy
      inv_sip[np.isinf(inv_sip)] = 0
      data['trk_dxy_sig_inverted'] = inv_sip
   data['training_out'] = -1.
   log_trkpt = np.log10(data.trk_pt)
   log_trkpt[np.isnan(log_trkpt)] = -9999
   data['log_trkpt'] = log_trkpt
   
   #apply pt-eta reweighting
   ## from hep_ml.reweight import GBReweighter
   ## from sklearn.externals import joblib
   ## reweighter = joblib.load('%s/%s_reweighting.pkl' % (mods, dataset))
   ## weights = reweighter.predict_weights(data[['trk_pt', 'trk_eta']])
   kmeans_model = '%s/kmeans_%s_weighter.pkl' % (mods, dataset)
   if not os.path.isfile(kmeans_model):
      print 'I could not find the appropriate model, using the general instead'
      print '!!! NOTA BENE: use .pkl not .plk !!!'
      kmeans_model = '%s/kmeans_%s_weighter.pkl' % (mods, tag)
   weights = kmeans_weighter(
      data[['log_trkpt', 'trk_eta']],
      kmeans_model
      )    
   data['weight'] = weights*np.invert(data.is_e) + data.is_e

   ## original_weight = HistWeighter('../data/fakesWeights.txt')
   ## data['original_weight'] = np.invert(data.is_e)*original_weight.get_weight(data.log_trkpt, data.trk_eta)+data.is_e

   #
   # pre-process data
   #   
   if 'trk_charge' in data.columns:
      for feat in ['ktf_ecal_cluster_dphi', 'ktf_hcal_cluster_dphi', 'preid_trk_ecal_Dphi']:
         if feat in data.columns:
            data[feat] = data[feat]*data['trk_charge']

#      charge = data.trk_charge
#      for feat in ['gsf_ecal_cluster_ematrix', 'ktf_ecal_cluster_ematrix']:
#         if feat in multi_dim:
#            multi_dim[feat][charge == -1] = np.flip(multi_dim[feat][charge == -1], axis=2)

   #add baseline seeding (for seeding only)
   if for_seeding:
      if 'preid_trk_ecal_match' in data.columns:
         data['baseline'] = (
            data.preid_trk_ecal_match | 
            (np.invert(data.preid_trk_ecal_match) & data.preid_trkfilter_pass & data.preid_mva_pass)
            )
      elif 'trk_pass_default_preid' in data.columns:
         data['baseline'] = data.trk_pass_default_preid
      else:
         data['baseline'] = False

   from features import labeling
   #convert bools to integers
   for c in data.columns:
      if data[c].dtype == np.dtype('bool') and c not in labeling:
         data[c] = data[c].astype(int)

   if not multi_dim:
      return data
   else:
      return data, multi_dim


def train_test_split(data, div, thr):
   mask = data.evt % div
   mask = mask < thr
   return data[mask], data[np.invert(mask)]
