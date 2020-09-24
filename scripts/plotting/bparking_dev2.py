from __future__ import print_function
from __future__ import absolute_import
import builtins
import future
from future.utils import raise_with_traceback
import past
import six

import matplotlib
matplotlib.use('Agg') # choose backend before doing anything else with pyplot! ('macosx')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from standalone import binomial_hpdr

################################################################################
# 
def bparking_dev2(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### BPARKING DEV2 ##########################################################')

   TRK_DENOM = True

   threshold = 1.e-1

   #############
   # ROC CURVE #
   #############

   plt.figure(figsize=(6,6))
   ax = plt.subplot(111)
   plt.title('KF tracks, pT > 0.5 GeV' if TRK_DENOM else 'Electrons, pT > 0.5 GeV')
   plt.xlim(1.e-5,1.)
   plt.ylim(0.,1.0)
   plt.gca().set_xscale('log')
   plt.xlabel('Mistag rate')
   plt.ylabel('Efficiency')
   ax.tick_params(axis='x', pad=10.)
   plt.grid(True)

   ########################################
   # "by chance" line

   plt.plot(np.arange(0.,1.,plt.xlim()[0]),
            np.arange(0.,1.,plt.xlim()[0]),
            ls='dotted',lw=0.5,color='gray',label="By chance")

   ########################################
   # Electron (pT > 0.5 GeV)

   has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
   has_ele = (test.has_ele) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
   has_obj = has_trk if TRK_DENOM else has_ele

   denom = has_obj&test.is_e; numer = has_ele&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_obj&(~test.is_e); numer = has_ele&denom;
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([ele_fr], [ele_eff],
            marker='o', markerfacecolor='blue', markeredgecolor='blue', 
            markersize=8,linestyle='none',
            label='Low-pT electron',
            )

   id_branch = 'gsf_bdtout1'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
   plt.plot(id_fpr*ele_fr, 
            id_tpr*ele_eff,
            linestyle='solid', color='red', linewidth=1.0,
            label='Seeding (AUC={:.3f})'.format(id_auc))
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_seed = test[id_branch]>id_score[idx]

   for threshold in [2.,3.,4.,5.,6.] :
      idx = np.abs(id_score-float(threshold)).argmin()
      wp = test[id_branch]>id_score[idx]
      x,y = id_fpr[idx]*ele_fr,id_tpr[idx]*ele_eff
      plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='red', markersize=6)
      plt.text(x, y+0.02, f"{threshold:.1f}", fontsize=6, ha='center', va='center', color='red' )

   id_branch = 'ele_mva_value'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
   plt.plot(id_fpr*ele_fr,
            id_tpr*ele_eff,
            linestyle='solid', color='blue', linewidth=1.0,
            label='2019Aug07 (AUC={:.3f})'.format(id_auc))
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_default = test[id_branch]>id_score[idx]

   for threshold in [2.,3.,4.,5.,6.] :
      idx = np.abs(id_score-float(threshold)).argmin()
      wp = test[id_branch]>id_score[idx]
      x,y = id_fpr[idx]*ele_fr,id_tpr[idx]*ele_eff
      plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='blue', markersize=6)
      plt.text(x, y+0.02, f"{threshold:.1f}", fontsize=6, ha='center', va='center', color='blue' )

   id_branch = 'ele_mva_value_depth13'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
   plt.plot(id_fpr*ele_fr, 
            id_tpr*ele_eff,
            linestyle='solid', color='orange', linewidth=1.0,
            label='depth=13 (AUC={:.3f})'.format(id_auc))
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_depth13 = test[id_branch]>id_score[idx]

   for threshold in [2.,3.,4.,5.,6.] :
      idx = np.abs(id_score-float(threshold)).argmin()
      wp = test[id_branch]>id_score[idx]
      x,y = id_fpr[idx]*ele_fr,id_tpr[idx]*ele_eff
      plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='orange', markersize=6)
      plt.text(x, y+0.02, f"{threshold:.1f}", fontsize=6, ha='center', va='center', color='orange' )

   id_branch = 'ele_mva_value_depth15'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
   plt.plot(id_fpr*ele_fr, 
            id_tpr*ele_eff,
            linestyle='solid', color='purple', linewidth=1.0,
            label='depth=15 (AUC={:.3f})'.format(id_auc))
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_depth15 = test[id_branch]>id_score[idx]

   for threshold in [2.,3.,4.,5.,6.] :
      idx = np.abs(id_score-float(threshold)).argmin()
      wp = test[id_branch]>id_score[idx]
      x,y = id_fpr[idx]*ele_fr,id_tpr[idx]*ele_eff
      plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='purple', markersize=6)
      plt.text(x, y+0.02, f"{threshold:.1f}", fontsize=6, ha='center', va='center', color='purple' )

   # PF electrons
   if TRK_DENOM :

      pf_has_trk = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
      pf_has_ele = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)

      denom = pf_has_trk&egamma.is_e; numer = pf_has_ele&denom
      pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
      denom = pf_has_trk&(~egamma.is_e); numer = pf_has_ele&denom
      pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
      plt.plot([pf_fr], [pf_eff],
               marker='o', color='green', 
               markersize=8, linestyle='none',
               label='PF electron')

      pf_id_branch = 'ele_mva_value_retrained'
      pf_id_fpr,pf_id_tpr,pf_id_score = roc_curve(egamma.is_e[pf_has_ele],egamma[pf_id_branch][pf_has_ele])
      pf_id_auc = roc_auc_score(egamma.is_e[pf_has_ele],egamma[pf_id_branch][pf_has_ele]) if len(set(egamma.is_e[pf_has_ele])) > 1 else 0.
      plt.plot(pf_id_fpr*pf_fr, 
               pf_id_tpr*pf_eff,
               linestyle='solid', color='green', linewidth=1.0,
               label='Retrained ID (AUC={:.3f})'.format(pf_id_auc))

      for threshold in [0.,1.,2.,3.] :
         idx = np.abs(pf_id_score-float(threshold)).argmin()
         wp = egamma[pf_id_branch]>pf_id_score[idx]
         x,y = pf_id_fpr[idx]*pf_fr,pf_id_tpr[idx]*pf_eff
         plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='green', markersize=6)
         plt.text(x, y+0.02, f"{threshold:.1f}", fontsize=6, ha='center', va='center', color='green' )

#   ########################################
#   # Electron (pT > 5.0 GeV)
#
#   has_high = has_ele & (test.trk_pt>5.0)
#   denom = has_obj&test.is_e; numer = has_high&denom;
#   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   denom = has_obj&(~test.is_e); numer = has_high&denom;
#   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   plt.plot([ele_fr], [ele_eff],
#            marker='^', markerfacecolor='none', markeredgecolor='blue', 
#            markersize=8,linestyle='none',
#            label='pT(trk) > 5.0 GeV',
#            )
#
#   id_branch = 'gsf_bdtout1'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
#   id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='red',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_unbiased_high = test[id_branch][has_high]>id_score[idx]
#
#   id_branch = 'ele_mva_value'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
#   id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='blue',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_default_high = test[id_branch][has_high]>id_score[idx]
#
#   id_branch = 'ele_mva_value_depth13'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
#   id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='orange',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_depth13_high = test[id_branch][has_high]>id_score[idx]
#
#   id_branch = 'ele_mva_value_depth15'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
#   id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='purple',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_depth15_high = test[id_branch][has_high]>id_score[idx]
#
#   #wp_default = wp_default_high
#   #wp_depth10 = wp_depth10_high
#   #wp_depth11 = wp_depth11_high
#   # PF electrons
#
#   ########################################
#   # Electron (2.0 < pT < 5.0 GeV)
#
#   has_med = has_ele & (test.trk_pt>2.0) & (test.trk_pt<5.0)
#   denom = has_obj&test.is_e; numer = has_med&denom;
#   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   denom = has_obj&(~test.is_e); numer = has_med&denom;
#   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   plt.plot([ele_fr], [ele_eff],
#            marker='s', markerfacecolor='none', markeredgecolor='blue', 
#            markersize=8,linestyle='none',
#            label='2.0 < pT(trk) < 5.0 GeV',
#            )
#
#   id_branch = 'gsf_bdtout1'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_med],test[id_branch][has_med])
#   id_auc = roc_auc_score(test.is_e[has_med],test[id_branch][has_med]) if len(set(test.is_e[has_med])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='red',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_unbiased_med = test[id_branch][has_med]>id_score[idx]
#
#   id_branch = 'ele_mva_value'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_med],test[id_branch][has_med])
#   id_auc = roc_auc_score(test.is_e[has_med],test[id_branch][has_med]) if len(set(test.is_e[has_med])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='blue',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_default_med = test[id_branch][has_med]>id_score[idx]
#
#   id_branch = 'ele_mva_value_depth13'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_med],test[id_branch][has_med])
#   id_auc = roc_auc_score(test.is_e[has_med],test[id_branch][has_med]) if len(set(test.is_e[has_med])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='orange',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_depth13_med = test[id_branch][has_med]>id_score[idx]
#
#   id_branch = 'ele_mva_value_depth15'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_med],test[id_branch][has_med])
#   id_auc = roc_auc_score(test.is_e[has_med],test[id_branch][has_med]) if len(set(test.is_e[has_med])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='purple',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_depth15_med = test[id_branch][has_med]>id_score[idx]
#
#   #wp_default = wp_default_med
#   #wp_depth10 = wp_depth10_med
#   #wp_depth11 = wp_depth11_med
#
#   ########################################
#   # Electron (0.5 < pT < 2.0 GeV)
#
#   has_low = has_ele & (test.trk_pt<2.0)
#   denom = has_obj&test.is_e; numer = has_low&denom;
#   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   denom = has_obj&(~test.is_e); numer = has_low&denom;
#   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   plt.plot([ele_fr], [ele_eff],
#            marker='v', markerfacecolor='none', markeredgecolor='blue', 
#            markersize=8,linestyle='none',
#            label='pT(trk) < 2.0 GeV',
#            )
#
#   id_branch = 'gsf_bdtout1'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
#   id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='red',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_unbiased_low = test[id_branch][has_low]>id_score[idx]
#
#   id_branch = 'ele_mva_value'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
#   id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='blue',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_default_low = test[id_branch][has_low]>id_score[idx]
#
#   id_branch = 'ele_mva_value_depth13'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
#   id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='orange',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_depth13_low = test[id_branch][has_low]>id_score[idx]
#
#   id_branch = 'ele_mva_value_depth15'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
#   id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid',color='purple',linewidth=1.0,
#            )
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_depth15_low = test[id_branch][has_low]>id_score[idx]
#
#   #wp_default = wp_default_low
#   #wp_depth10 = wp_depth10_low
#   #wp_depth11 = wp_depth11_low
#
#   if TRK_DENOM :
#
#      pf_has_trk = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
#      pf_has_ele = (egamma.has_ele) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
#
#      pf_has_high = pf_has_ele & (egamma.trk_pt>5.0)
#      denom = pf_has_trk&egamma.is_e; numer = pf_has_high&denom
#      pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#      denom = pf_has_trk&(~egamma.is_e); numer = pf_has_high&denom
#      pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#      plt.plot([pf_fr], [pf_eff],
#               marker='^', markerfacecolor='none', markeredgecolor='green', 
#               markersize=8,linestyle='none',
#               label='pT(trk) > 5.0 GeV',
#               )
#      pf_id_branch = 'ele_mva_value_retrained'
#      pf_id_fpr,pf_id_tpr,pf_id_score = roc_curve(egamma.is_e[pf_has_high],egamma[pf_id_branch][pf_has_high])
#      pf_id_auc = roc_auc_score(egamma.is_e[pf_has_high],egamma[pf_id_branch][pf_has_high]) if len(set(egamma.is_e[pf_has_high])) > 1 else 0.
#      plt.plot(pf_id_fpr*pf_fr, 
#               pf_id_tpr*pf_eff,
#               linestyle='solid',color='green',linewidth=1.0,
#               )
#
#      pf_has_med = pf_has_ele & (egamma.trk_pt>2.0) & (egamma.trk_pt<5.0)
#      denom = pf_has_trk&egamma.is_e; numer = pf_has_med&denom
#      pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#      denom = pf_has_trk&(~egamma.is_e); numer = pf_has_med&denom
#      pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#      plt.plot([pf_fr], [pf_eff],
#               marker='s', markerfacecolor='none', markeredgecolor='green', 
#               markersize=8,linestyle='none',
#               label='2.0 < pT(trk) < 5.0 GeV',
#               )
#      pf_id_branch = 'ele_mva_value_retrained'
#      pf_id_fpr,pf_id_tpr,pf_id_score = roc_curve(egamma.is_e[pf_has_med],egamma[pf_id_branch][pf_has_med])
#      pf_id_auc = roc_auc_score(egamma.is_e[pf_has_med],egamma[pf_id_branch][pf_has_med]) if len(set(egamma.is_e[pf_has_med])) > 1 else 0.
#      plt.plot(pf_id_fpr*pf_fr, 
#               pf_id_tpr*pf_eff,
#               linestyle='solid',color='green',linewidth=1.0,
#               )
#
#      pf_has_low = pf_has_ele & (egamma.trk_pt<2.0)
#      denom = pf_has_trk&egamma.is_e; numer = pf_has_low&denom
#      pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#      denom = pf_has_trk&(~egamma.is_e); numer = pf_has_low&denom
#      pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#      plt.plot([pf_fr], [pf_eff],
#               marker='v', markerfacecolor='none', markeredgecolor='green', 
#               markersize=8,linestyle='none',
#               label='pT(trk) < 2.0 GeV',
#               )
#      pf_id_branch = 'ele_mva_value_retrained'
#      pf_id_fpr,pf_id_tpr,pf_id_score = roc_curve(egamma.is_e[pf_has_low],egamma[pf_id_branch][pf_has_low])
#      pf_id_auc = roc_auc_score(egamma.is_e[pf_has_low],egamma[pf_id_branch][pf_has_low]) if len(set(egamma.is_e[pf_has_low])) > 1 else 0.
#      plt.plot(pf_id_fpr*pf_fr, 
#               pf_id_tpr*pf_eff,
#               linestyle='solid',color='green',linewidth=1.0,
#               )
   
   ##########
   # Finish up ... 
   plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False) # 'center right'
   plt.tight_layout()
   #plt.savefig(dir+'/roc.pdf')
   #plt.gca().set_xscale('log')
   plt.savefig(dir+'/roc_log.pdf')
   #plt.gca().set_yscale('log')
   #plt.savefig(dir+'/roc_loglog.pdf')
   plt.clf()
   plt.close()
