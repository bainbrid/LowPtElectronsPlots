import uproot
import numpy as np
import numba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf = uproot.open('../run/EleGun.root')
feats = tf['features/tree'].arrays([
      'gsf_pt', 'gsf_eta', 'gsf_charge', 
      'preid_bdtout1', 'preid_bdtout2', 'is_e',
      'preid_mva1_pass', 'preid_mva2_pass',
      'gsf_extapolated_eta', 'gsf_extapolated_phi', 
      'sc_cluster_eta', 'sc_cluster_phi'
      ])


class JaggedAccumulator(object):
   def __init__(self):
      self.ret_ = []
      self.ret_strt_ = []
      self.ret_stop_ = []
   
   def add_entry(self, arr):
      self.ret_strt_.append(len(self.ret_))
      self.ret_.extend(arr)
      self.ret_stop_.append(len(self.ret_))

   def to_jagged(self):
      return uproot.interp.jagged.JaggedArray(
         np.array(self.ret_     ),
         np.array(self.ret_strt_),
         np.array(self.ret_stop_),
         )

@numba.jit
def jagged_masking(jag, mask):
   ret = JaggedAccumulator()
   for entry, imask in zip(jag, mask):
      if imask:
         ret.add_entry(entry)
   return ret.to_jagged()

def mask_dict(dd, mask):
   ret = {}
   for name, arr in dd.iteritems():
      if isinstance(arr, np.ndarray):
         ret[name] = arr[mask]
      else:
         ret[name] = jagged_masking(arr, mask)
   return ret

@numba.jit
def get_deta_dphi(gsf_etas, gsf_phis, cl_etas, cl_phis):
   delta_rs = JaggedAccumulator()
   delta_etas = JaggedAccumulator()
   delta_phis = JaggedAccumulator()
   for gsf_eta, gsf_phi, cl_eta, cl_phi in zip(gsf_etas, gsf_phis, cl_etas, cl_phis):      
      points = zip(gsf_eta, gsf_phi)
      clusters = zip(cl_eta, cl_phi)
      dr_min = [float('inf')]*len(clusters)
      match  = [-1]*len(clusters)
      for i_cl, cluster in enumerate(clusters):
         dr = float('inf')
         idx = -1
         for i_p, point in enumerate(points):
            dr2 = np.sqrt((cluster[0] - point[0])**2 + (cluster[1] - point[1])**2)
            if dr2 < dr:
               dr = dr2
               idx = i_p
         if dr < dr_min[i_cl]:
            dr_min[i_cl] = dr
            match[i_cl] = idx
      
      deta = []
      dphi = []
      for ipoint, cluster in zip(match, clusters):
         deta.append(cluster[0] - points[ipoint][0])
         dphi.append(cluster[1] - points[ipoint][1])

      delta_rs.add_entry(dr_min)
      delta_etas.add_entry(deta)
      delta_phis.add_entry(dphi)

   return delta_rs.to_jagged(), \
      delta_etas.to_jagged(), \
      delta_phis.to_jagged()
      
@numba.jit
def jagged_mult(jag, norm):
   ret = JaggedAccumulator()
   for jj, ii in zip(jag, norm):
      ret.add_entry(list(ii*jj))
   return ret.to_jagged()
   
def jagged_len(jag):
   return np.array([len(i) for i in jag])

def jagged_min(jag):
   return np.array([np.abs(i).min() for i in jag])

def jagged_max(jag):
   return np.array([np.abs(i).max() for i in jag])

mm = ((feats['gsf_pt'] > 0) & feats['is_e'] & (feats['preid_mva1_pass'] | feats['preid_mva2_pass']))
mm &= ((feats['gsf_extapolated_eta'].stops - feats['gsf_extapolated_eta'].starts) > 1)
feats_back = feats
feats = mask_dict(feats, mm)

drs, detas, dphis = get_deta_dphi(
   feats['gsf_extapolated_eta'], feats['gsf_extapolated_phi'],
   feats['sc_cluster_eta'], feats['sc_cluster_phi']
   )

def make_op(jj, op):
   if op == '_min':
      return jagged_min(jj)
   elif op == '_max':
      return jagged_max(jj)
   else:
      return jj.content

for op in ['', '_min', '_max']:
   plt.figure(figsize=[8,8])
   plt.hist(make_op(drs, op), bins=50, histtype='stepfilled')
   plt.axvline(1, color='r')
   plt.axvline(0.5, color='r', linestyle='--')
   plt.xlabel('DR')
   plt.gca().set_yscale('log')
   plt.savefig('matching_dR%s.png' % op)
   plt.clf()   
   plt.hist(make_op(detas, op), bins=50, histtype='stepfilled')
   plt.xlabel('DEta')
   plt.gca().set_yscale('log')
   plt.savefig('matching_dEta%s.png' % op)
   plt.clf()   
   plt.hist(make_op(dphis, op), bins=50, histtype='stepfilled')
   plt.xlabel('DPhi')
   plt.gca().set_yscale('log')
   plt.savefig('matching_dPhi%s.png' % op)
   plt.clf()

plt.hist(jagged_len(drs), bins=50, histtype='stepfilled')
plt.xlabel('# clusters')
plt.gca().set_yscale('log')
plt.savefig('nmatching.png')
plt.clf()

dphi_ch = jagged_mult(dphis, feats['gsf_charge'])
plt.hist(dphi_ch.content, bins=50, histtype='stepfilled')
plt.xlabel('DPhi * charge')
plt.gca().set_yscale('log')
plt.savefig('matching_dPhi_charge.png')
plt.clf()

def jagged_thr(jag, thr):
   return np.array([(np.abs(i) < thr).any() for i in jag])

thrs = np.arange(0, 7, 0.05)
n_entries = float(feats['is_e'].shape[0])
dr   = np.array([jagged_thr(drs, i).sum() for i in thrs])/n_entries
plt.plot(thrs, dr, 'r-')
plt.xlabel('DPhi * charge')
plt.xlim(0.05, 7)
plt.gca().set_xscale('log')
plt.savefig('thr_eff.png')
plt.clf()
##ellipse = []np.array([jagged_thr(deta, i).sum() for i in thrs])/n_entries
##rectangule = []np.array([jagged_thr(dphi, i).sum() for i in thrs])/n_entries
##for thr in thrs:
##   dr.append(jagged_thr(drs, i).sum()/n_entries)
   
