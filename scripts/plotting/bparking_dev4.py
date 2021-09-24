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
def bparking_dev4(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### BPARKING DEV4 ##########################################################')

   inclusive = False
   TRK_DENOM = True

   #############
   # ROC CURVE #
   #############

   plt.figure(figsize=(6,6))
   ax = plt.subplot(111)
   plt.title('KF tracks, pT > 0.5 GeV' if TRK_DENOM else 'Electrons, pT > 0.5 GeV')
   plt.xlim(1.e-5,1.)
   plt.ylim(0.,1.0 if inclusive else 0.6)
   plt.gca().set_xscale('log')
   plt.xlabel('Mistag rate')
   plt.ylabel('Efficiency')
   ax.tick_params(axis='x', pad=10.)
   plt.grid(True)

   ########################################
   # plotting config
   offset = 0.015 * (plt.ylim()[1]-plt.ylim()[0])
   msl=4 # markersize large
   mss=3 # markersize small
   fs=4  # fontsize

   ########################################
   # "by chance" line

   plt.plot(np.arange(0.,1.,plt.xlim()[0]),
            np.arange(0.,1.,plt.xlim()[0]),
            ls='dotted',lw=0.5,color='gray',label="By chance")

   ########################################
   # INIT

   has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
   has_ele = (test.has_ele) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
   has_obj = has_trk if TRK_DENOM else has_ele
   print(len(has_obj),has_obj.sum())
   print(has_obj)

   ########################################
   # Electron (pT > 0.5 GeV)

   if inclusive :

      denom = has_obj&test.is_e; numer = has_ele&denom;
      eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
      denom = has_obj&(~test.is_e); numer = has_ele&denom;
      fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
      plt.plot([fr], [eff],
               marker='o', markerfacecolor='blue', markeredgecolor='blue', 
               markersize=8,linestyle='none',
               label='Low-pT electron',
               )
   
      id_branch = 'ele_mva_value_depth15'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
      id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
      plt.plot(id_fpr*fr, 
               id_tpr*eff,
               linestyle='solid', color='purple', linewidth=1.0,
               label='2020Sep15 (AUC={:.3f})'.format(id_auc))
   
      colour = 'purple'
      for threshold in np.arange(2,17) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
      id_branch = 'ele_mva_value'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
      id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
      plt.plot(id_fpr*fr,
               id_tpr*eff,
               linestyle='solid', color='blue', linewidth=1.0,
               label='2019Aug07 (AUC={:.3f})'.format(id_auc))
   
      colour = 'blue'
      for threshold in np.arange(2,14) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
      id_branch = 'ele_mva_value_old'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
      id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
      plt.plot(id_fpr*fr,
               id_tpr*eff,
               linestyle='solid', color='orange', linewidth=1.0,
               label='2019Jul22 (AUC={:.3f})'.format(id_auc))
   
      colour = 'orange'
      for threshold in np.arange(2,13) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
      id_branch = 'gsf_bdtout1'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
      id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
      plt.plot(id_fpr*fr, 
               id_tpr*eff,
               linestyle='solid', color='red', linewidth=1.0,
               label='Seeding (AUC={:.3f})'.format(id_auc))
   
      colour = 'red'
      for threshold in np.arange(2,17) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)

      # PF electrons
      if TRK_DENOM :
   
         has_trk = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
         has_ele = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
   
         denom = has_trk&egamma.is_e; numer = has_ele&denom
         eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
         denom = has_trk&(~egamma.is_e); numer = has_ele&denom
         fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
         plt.plot([fr], [eff],
                  marker='o', color='green', 
                  markersize=8, linestyle='none',
                  label='PF electron')
   
         id_branch = 'ele_mva_value_retrained'
         id_fpr,id_tpr,id_score = roc_curve(egamma.is_e[has_ele],egamma[id_branch][has_ele])
         id_auc = roc_auc_score(egamma.is_e[has_ele],egamma[id_branch][has_ele]) if len(set(egamma.is_e[has_ele])) > 1 else 0.
         plt.plot(id_fpr*fr, 
                  id_tpr*eff,
                  linestyle='solid', color='green', linewidth=1.0,
                  label='Retrained ID (AUC={:.3f})'.format(id_auc))
   
         colour = 'green'
         for threshold in np.arange(-1,8) :
            idx = np.abs(id_score-float(threshold/2.)).argmin()
            wp = test[id_branch]>id_score[idx]
            x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
            if threshold%2 == 0 :
               plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
               plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
            else :
               plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)

   else :

      ########################################
      # Electron (pT > 2.0 GeV)
   
      has_high = has_ele & (test.trk_pt>2.0)
      denom = has_obj&test.is_e; numer = has_high&denom;
      eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
      denom = has_obj&(~test.is_e); numer = has_high&denom;
      fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
      plt.plot([fr], [eff],
               marker='^', markerfacecolor='blue', markeredgecolor='blue',
               markersize=8,linestyle='none',
               label='pT(trk) > 2.0 GeV',
               )
      print("pT>2.0:",eff,fr)
   
      id_branch = 'ele_mva_value_depth15'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
      id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
      plt.plot(id_fpr*fr,
               id_tpr*eff,
               linestyle='solid',color='purple',linewidth=1.0,
               )
   
      colour = 'purple'
      for threshold in [-12,-8,-4,0,2,4,6,8,10,12,14,16] :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
      id_branch = 'ele_mva_value'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
      id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
      plt.plot(id_fpr*fr,
               id_tpr*eff,
               linestyle='solid',color='blue',linewidth=1.0,
               )
   
      colour = 'blue'
      for threshold in np.arange(2,13) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
            
      id_branch = 'ele_mva_value_old'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
      id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
      plt.plot(id_fpr*fr,
               id_tpr*eff,
               linestyle='solid',color='orange',linewidth=1.0,
               )
   
      colour = 'orange'
      for threshold in np.arange(2,13) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
      id_branch = 'gsf_bdtout1'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
      id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
      plt.plot(id_fpr*fr,
               id_tpr*eff,
               linestyle='solid',color='red',linewidth=1.0,
               )
   
      colour = 'red'
      for threshold in np.arange(2,17) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
      ########################################
      # Electron (0.5 < pT < 2.0 GeV)
   
      has_low = has_ele & (test.trk_pt<2.0)
      denom = has_obj&test.is_e; numer = has_low&denom;
      eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
      denom = has_obj&(~test.is_e); numer = has_low&denom;
      fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
      plt.plot([fr], [eff],
               marker='v', markerfacecolor='blue', markeredgecolor='blue', 
               markersize=8,linestyle='none',
               label='pT(trk) < 2.0 GeV',
               )
      print("pT<2.0:",eff,fr)
   
      id_branch = 'ele_mva_value_depth15'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
      id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
      plt.plot(id_fpr*fr,
               id_tpr*eff,
               linestyle='solid',color='purple',linewidth=1.0,
               )
   
      colour = 'purple'
      for threshold in np.arange(2,13) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
      id_branch = 'ele_mva_value'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
      id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
      plt.plot(id_fpr*fr,
               id_tpr*eff,
               linestyle='solid',color='blue',linewidth=1.0,
               )
   
      colour = 'blue'
      for threshold in np.arange(2,13) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
      id_branch = 'ele_mva_value_old'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
      id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
      plt.plot(id_fpr*fr,
               id_tpr*eff,
               linestyle='solid',color='orange',linewidth=1.0,
               )
   
      colour = 'orange'
      for threshold in np.arange(2,13) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
      id_branch = 'gsf_bdtout1'
      id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
      id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
      plt.plot(id_fpr*fr,
               id_tpr*eff,
               linestyle='solid',color='red',linewidth=1.0,
               )
   
      colour = 'red'
      for threshold in np.arange(2,13) :
         idx = np.abs(id_score-float(threshold/2.)).argmin()
         wp = test[id_branch]>id_score[idx]
         x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
         if threshold%2 == 0 :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
            plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
         else :
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
      if TRK_DENOM :
   
         has_trk = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
         has_ele = (egamma.has_ele) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
   
         print(len(has_trk),has_trk.sum())
         print(has_trk)
   
         has_high = has_ele & (egamma.trk_pt>2.0)
         denom = has_trk&egamma.is_e; numer = has_high&denom
         eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
         denom = has_trk&(~egamma.is_e); numer = has_high&denom
         fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
         plt.plot([fr], [eff],
                  marker='^', markerfacecolor='green', markeredgecolor='green', 
                  markersize=8,linestyle='none',
                  label='pT(trk) > 2.0 GeV',
                  )
   
         id_branch = 'ele_mva_value_retrained'
         id_fpr,id_tpr,id_score = roc_curve(egamma.is_e[has_high],egamma[id_branch][has_high])
         id_auc = roc_auc_score(egamma.is_e[has_high],egamma[id_branch][has_high]) if len(set(egamma.is_e[has_high])) > 1 else 0.
         plt.plot(id_fpr*fr, 
                  id_tpr*eff,
                  linestyle='solid',color='green',linewidth=1.0,
                  )
   
         colour = 'green'
         for threshold in np.arange(-1,8) :
            idx = np.abs(id_score-float(threshold/2.)).argmin()
            wp = test[id_branch]>id_score[idx]
            x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
            if threshold%2 == 0 :
               plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=msl)
               plt.text(x, y+offset, f"{threshold/2.:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color=colour )
            else :
               plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor=colour, markersize=mss)
   
         for threshold in [0.,1.,2.,3.] :
            idx = np.abs(id_score-float(threshold)).argmin()
            wp = egamma[id_branch]>id_score[idx]
            x,y = id_fpr[idx]*fr,id_tpr[idx]*eff
            plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='green', markersize=msl)
            plt.text(x, y+offset, f"{threshold:.1f}", fontsize=fs, ha='center', va='center', clip_on=True, color='green' )
   
         has_low = has_ele & (egamma.trk_pt<2.0)
         denom = has_trk&egamma.is_e; numer = has_low&denom
         eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
         denom = has_trk&(~egamma.is_e); numer = has_low&denom
         fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
         plt.plot([fr], [eff],
                  marker='v', markerfacecolor='green', markeredgecolor='green', 
                  markersize=8,linestyle='none',
                  label='pT(trk) < 2.0 GeV',
                  )
   
         id_branch = 'ele_mva_value_retrained'
         id_fpr,id_tpr,id_score = roc_curve(egamma.is_e[has_low],egamma[id_branch][has_low])
         id_auc = roc_auc_score(egamma.is_e[has_low],egamma[id_branch][has_low]) if len(set(egamma.is_e[has_low])) > 1 else 0.
         plt.plot(id_fpr*fr, 
                  id_tpr*eff,
                  linestyle='solid',color='green',linewidth=1.0,
                  )
   
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
