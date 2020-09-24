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

################################################################################
# 
def seed(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### SEED ###########################################################')

   #############
   # ROC CURVE #
   #############

   plt.figure()
   ax = plt.subplot(111)
   plt.title('Effciency and mistag rate w.r.t. KF tracks')
   plt.xlim(1.e-4,1.1)
   plt.ylim([0., 1.03])
   plt.xlabel('FPR')
   plt.ylabel('TPR')
   ax.tick_params(axis='x', pad=10.)
   plt.gca().set_xscale('log')

   ##########
   # "by chance" line
   plt.plot(np.arange(0.,1.,plt.xlim()[0]),np.arange(0.,1.,plt.xlim()[0]),'k--',lw=0.5)

   ##########
   # Low-pT GSF tracks + unbiased seed BDT 
   has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   if AxE is True :
      unb_eff = float((has_gsf&test.is_e).sum()) / float((test.is_e).sum()) \
          if float((test.is_e).sum()) > 0. else 0.
   else :
      unb_eff = float((has_gsf&has_trk&test.is_e).sum()) / float((has_trk&test.is_e).sum()) \
          if float((has_trk&test.is_e).sum()) > 0. else 0.
   unb_fr = float((has_gsf&(~test.is_e)&has_trk).sum()) / float(((~test.is_e)&has_trk).sum()) \
       if float(((~test.is_e)&has_trk).sum()) > 0. else 0.
   unb_fpr,unb_tpr,unb_score = roc_curve(test.is_e[has_gsf],test.gsf_bdtout1[has_gsf])
   unb_auc = roc_auc_score(test.is_e[has_gsf],test.gsf_bdtout1[has_gsf]) if len(set(test.is_e[has_gsf])) > 1 else 0.
   plt.plot(unb_fpr*unb_fr, unb_tpr*unb_eff,
            linestyle='solid', color='red', linewidth=1.,
            label='Low-pT GSF track + unbiased seed BDT, AUC={:.3f}'.format(unb_auc))
   plt.plot([unb_fr], [unb_eff],
            marker='o',color='red', markersize=8, linestyle='None', clip_on=False)
   
   ##########
   # Low-pT GSF tracks + biased seed BDT 
   has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   if AxE is True :
      b_eff = float((has_gsf&test.is_e).sum()) / float((test.is_e).sum()) \
          if float((test.is_e).sum()) > 0. else 0.
   else :
      b_eff = float((has_gsf&has_trk&test.is_e).sum()) / float((has_trk&test.is_e).sum()) \
          if float((has_trk&test.is_e).sum()) > 0. else 0.
   b_fr = float((has_gsf&(~test.is_e)&has_trk).sum()) / float(((~test.is_e)&has_trk).sum()) \
       if float(((~test.is_e)&has_trk).sum()) > 0. else 0.
   b_fpr,b_tpr,b_score = roc_curve(test.is_e[has_gsf],test.gsf_bdtout2[has_gsf])
   b_auc = roc_auc_score(test.is_e[has_gsf],test.gsf_bdtout2[has_gsf]) if len(set(test.is_e[has_gsf])) > 1 else 0.
   plt.plot(b_fpr*b_fr, b_tpr*b_eff,
            linestyle='dashed', color='red', linewidth=0.5,
            label='Low-pT GSF track + biased seed BDT, AUC={:.3f}'.format(b_auc))
   plt.plot([b_fr], [b_eff],
            marker='o',color='red', markersize=8, linestyle='None', clip_on=False)
   
   ##########
   # EGamma GSF tracks and PF GSF electrons
   has_trk = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
   has_gsf = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_pfgsf = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.5)
   if has_pfgsf_branches is False : has_pfgsf = has_gsf #@@ HACK
   has_ele = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)

   if AxE is True :
      eg_eff = float((has_pfgsf&egamma.is_e).sum()) / float((egamma.is_e).sum()) \
          if float((egamma.is_e).sum()) > 0. else 0.
   else :
      eg_eff = float((has_pfgsf&has_trk&egamma.is_e).sum()) / float((has_trk&egamma.is_e).sum()) \
          if float((has_trk&egamma.is_e).sum()) > 0. else 0.
   eg_fr = float((has_pfgsf&(~egamma.is_e)&has_trk).sum()) / float(((~egamma.is_e)&has_trk).sum()) \
       if float(((~egamma.is_e)&has_trk).sum()) > 0. else 0.
   plt.plot([eg_fr], [eg_eff],
            marker='o', color='green', markersize=8, linestyle='None',
            label='EGamma GSF track')

   if AxE is True :
      pf_eff = float((has_ele&egamma.is_e).sum()) / float((egamma.is_e).sum()) \
          if float((egamma.is_e).sum()) > 0. else 0.
   else :
      pf_eff = float((has_ele&has_trk&egamma.is_e).sum()) / float((has_trk&egamma.is_e).sum()) \
          if float((has_trk&egamma.is_e).sum()) > 0. else 0.
   pf_fr = float((has_ele&(~egamma.is_e)&has_trk).sum()) / float(((~egamma.is_e)&has_trk).sum()) \
       if float(((~egamma.is_e)&has_trk).sum()) > 0. else 0.
   plt.plot([pf_fr], [pf_eff],
            marker='o', color='purple', markersize=8, linestyle='None',
            label='PF GSF electron')

#   print('(has_trk).sum()',(has_trk).sum())
#   print('(has_gsf).sum()',(has_gsf).sum())
#   print('(has_pfgsf).sum()',(has_pfgsf).sum())
#   print('(has_ele).sum()',(has_ele).sum())
#
#   print('(has_gsf&has_trk).sum()',(has_gsf&has_trk).sum())
#   print('(has_pfgsf&has_trk).sum()',(has_pfgsf&has_trk).sum())
#   print('(has_ele&has_trk).sum()',(has_ele&has_trk).sum())
#
#   print('(egamma.is_e).sum()',(egamma.is_e).sum())
#   print('(has_trk&egamma.is_e).sum()',(has_trk&egamma.is_e).sum())
#
#   print('(has_gsf&has_trk&egamma.is_e).sum()',(has_gsf&has_trk&egamma.is_e).sum())
#   print('(has_pfgsf&has_trk&egamma.is_e).sum()',(has_pfgsf&has_trk&egamma.is_e).sum())
#   print('(has_ele&has_trk&egamma.is_e).sum()',(has_ele&has_trk&egamma.is_e).sum())
#
#   print('(has_gsf&egamma.is_e).sum()',(has_gsf&egamma.is_e).sum())
#   print('(has_pfgsf&egamma.is_e).sum()',(has_pfgsf&egamma.is_e).sum())
#   print('(has_ele&egamma.is_e).sum()',(has_ele&egamma.is_e).sum())
#
#   print('(~egamma.is_e).sum()',(~egamma.is_e).sum())
#   print('((~egamma.is_e)&has_trk).sum()',((~egamma.is_e)&has_trk).sum())
#   print('(has_gsf&(~egamma.is_e)&has_trk).sum()',(has_gsf&(~egamma.is_e)&has_trk).sum())
#   print('(has_pfgsf&(~egamma.is_e)&has_trk).sum()',(has_pfgsf&(~egamma.is_e)&has_trk).sum())
#   print('(has_ele&(~egamma.is_e)&has_trk).sum()',(has_ele&(~egamma.is_e)&has_trk).sum())

   # EGamma GSF tracks (ECAL-driven)
   has_trk |= egamma.seed_ecal_driven
   has_pfgsf &= egamma.seed_ecal_driven
   if AxE is True :
      _eff = float((has_pfgsf&egamma.is_e).sum()) / float((egamma.is_e).sum()) \
          if float((egamma.is_e).sum()) > 0. else 0.
   else :
      _eff = float((has_pfgsf&has_trk&egamma.is_e).sum()) / float((has_trk&egamma.is_e).sum()) \
          if float((has_trk&egamma.is_e).sum()) > 0. else 0.
   _fr = float((has_pfgsf&(~egamma.is_e)&has_trk).sum()) / float(((~egamma.is_e)&has_trk).sum()) \
       if float(((~egamma.is_e)&has_trk).sum()) > 0. else 0.
   plt.plot([_fr], [_eff],
            marker='o', markerfacecolor='none', markeredgecolor='green', markersize=8, linestyle='None',
            label='EGamma GSF track (ECAL-driven)')

   print('eff: {:.3f}, mistag: {:.4f}'.format(unb_eff,unb_fr),'Low-pT GSF tracks + unbiased seed BDT')
   print('eff: {:.3f}, mistag: {:.4f}'.format(b_eff,b_fr),'Low-pT GSF tracks + biased seed BDT')
   print('eff: {:.3f}, mistag: {:.4f}'.format(eg_eff,eg_fr),'EGamma GSF tracks')
   print('eff: {:.3f}, mistag: {:.4f}'.format(pf_eff,pf_fr),'EGamma GSF electrons')
   print('eff: {:.3f}, mistag: {:.4f}'.format(_eff,_fr),'EGamma GSF tracks (ECAL-driven)')

   ##########
   # Working points, "newly tuned"
   unb_VL = np.abs(unb_fpr*unb_fr-0.1).argmin()
   unb_L  = np.abs(unb_fpr*unb_fr-eg_fr*10.).argmin()
   unb_M  = np.abs(unb_fpr*unb_fr-eg_fr*3.).argmin()
   unb_T  = np.abs(unb_fpr*unb_fr-eg_fr).argmin()
   unb_VT = np.abs(unb_tpr*unb_eff-eg_eff).argmin()
   unb_PF = np.abs(unb_tpr*unb_eff-pf_eff).argmin()
   print("EGamma GSF track:  FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(eg_fr,eg_eff,np.nan))
   print("VLoose   (10% FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(unb_fpr[unb_VL]*unb_fr, unb_tpr[unb_VL]*unb_eff, unb_score[unb_VL]))
   print("Loose    (x10 FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(unb_fpr[unb_L]*unb_fr,  unb_tpr[unb_L]*unb_eff,  unb_score[unb_L]))
   print("Medium    (x3 FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(unb_fpr[unb_M]*unb_fr,  unb_tpr[unb_M]*unb_eff,  unb_score[unb_M]))
   print("Tight   (same FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(unb_fpr[unb_T]*unb_fr,  unb_tpr[unb_T]*unb_eff,  unb_score[unb_T]))
   print("VTight (same eff): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(unb_fpr[unb_VT]*unb_fr, unb_tpr[unb_VT]*unb_eff, unb_score[unb_VT]))
   print("PFELE  (same eff): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(unb_fpr[unb_PF]*unb_fr, unb_tpr[unb_PF]*unb_eff, unb_score[unb_PF]))
   plt.plot([unb_fpr[unb_VL]*unb_fr], [unb_tpr[unb_VL]*unb_eff], marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([unb_fpr[unb_L]*unb_fr],  [unb_tpr[unb_L]*unb_eff],  marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([unb_fpr[unb_M]*unb_fr],  [unb_tpr[unb_M]*unb_eff],  marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([unb_fpr[unb_T]*unb_fr],  [unb_tpr[unb_T]*unb_eff],  marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([unb_fpr[unb_VT]*unb_fr], [unb_tpr[unb_VT]*unb_eff], marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([unb_fpr[unb_PF]*unb_fr], [unb_tpr[unb_PF]*unb_eff], marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)

   # Original unbiased WPs
   gsf_VL = np.abs(unb_score-0.19).argmin()
   gsf_L  = np.abs(unb_score-1.20).argmin()
   gsf_M  = np.abs(unb_score-2.02).argmin()
   gsf_T  = np.abs(unb_score-3.05).argmin()
   plt.plot([unb_fpr[gsf_VL]*unb_fr], [unb_tpr[gsf_VL]*unb_eff], marker='^', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([unb_fpr[gsf_L]*unb_fr],  [unb_tpr[gsf_L]*unb_eff],  marker='^', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([unb_fpr[gsf_M]*unb_fr],  [unb_tpr[gsf_M]*unb_eff],  marker='^', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([unb_fpr[gsf_T]*unb_fr],  [unb_tpr[gsf_T]*unb_eff],  marker='^', markerfacecolor='none', markeredgecolor='k', markersize=4)
   # Original biased WPs
   gsf_VL = np.abs(b_score-(-1.99)).argmin()
   gsf_L  = np.abs(b_score-0.01).argmin()
   gsf_M  = np.abs(b_score-1.29).argmin()
   gsf_T  = np.abs(b_score-2.42).argmin()
   plt.plot([b_fpr[gsf_VL]*b_fr], [b_tpr[gsf_VL]*b_eff], marker='v', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([b_fpr[gsf_L]*b_fr],  [b_tpr[gsf_L]*b_eff],  marker='v', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([b_fpr[gsf_M]*b_fr],  [b_tpr[gsf_M]*b_eff],  marker='v', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([b_fpr[gsf_T]*b_fr],  [b_tpr[gsf_T]*b_eff],  marker='v', markerfacecolor='none', markeredgecolor='k', markersize=4)

   ##########
   # Low-pT GSF electrons + ID
   has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
   if AxE is True :
      id_eff = float((has_ele&test.is_e).sum()) / float((test.is_e).sum()) \
          if float((test.is_e).sum()) > 0. else 0.
   else :
      id_eff = float((has_ele&has_trk&test.is_e).sum()) / float((has_trk&test.is_e).sum()) \
          if float((has_trk&test.is_e).sum()) > 0. else 0.
   id_fr = float((has_ele&(~test.is_e)&has_trk).sum()) / float(((~test.is_e)&has_trk).sum()) \
       if float(((~test.is_e)&has_trk).sum()) > 0. else 0.
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test['training_out'][has_ele])
   id_auc = roc_auc_score(test.is_e[has_ele],test['training_out'][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
   plt.plot(id_fpr*id_fr, id_tpr*id_eff,
            linestyle='solid', color='blue', linewidth=1.,
            label='Low-pT GSF electron + ID, AUC={:.3f}'.format(id_auc))
   plt.plot([id_fr], [id_eff],
            marker='o',color='blue', markersize=8, linestyle=None)
   print('eff: {:.3f}, mistag: {:.4f}'.format(id_eff,id_fr),'Low-pT GSF electrons + ID')
   
   #########
   # Debug ...

   has_trk = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
   has_gsf = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_pfgsf = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.5)
   has_ele = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
   print("DEBUG",
         (has_trk).sum(),
         (has_gsf).sum(),
         (has_pfgsf).sum(),
         (has_ele).sum(),
         (egamma.is_e).sum(),
         (has_gsf&egamma.is_e).sum(),
         (has_pfgsf&egamma.is_e).sum(),
         (has_ele&egamma.is_e).sum(),
         '{:.3f}'.format(eg_eff),
         '{:.3f}'.format(pf_eff),
         (~egamma.is_e).sum(),
         (has_trk&(~egamma.is_e)).sum(),
         (has_gsf&has_trk&(~egamma.is_e)).sum(),
         (has_pfgsf&has_trk&(~egamma.is_e)).sum(),
         (has_ele&has_trk&(~egamma.is_e)).sum(),
         '{:.4f}'.format(eg_fr),
         '{:.4f}'.format(pf_fr),
         )

   has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_pfgsf = (test.has_pfgsf) & (test.pfgsf_pt>0.5) & (np.abs(test.pfgsf_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
   print("DEBUG",
         (has_trk).sum(),
         (has_gsf).sum(),
         (has_pfgsf).sum(),
         (has_ele).sum(),
         (test.is_e).sum(),
         (has_gsf&test.is_e).sum(),
         (has_pfgsf&test.is_e).sum(),
         (has_ele&test.is_e).sum(),
         '{:.3f}'.format(unb_eff),
         '{:.3f}'.format(id_eff),
         (~test.is_e).sum(),
         (has_trk&(~test.is_e)).sum(),
         (has_gsf&has_trk&(~test.is_e)).sum(),
         (has_pfgsf&has_trk&(~test.is_e)).sum(),
         (has_ele&has_trk&(~test.is_e)).sum(),
         '{:.4f}'.format(unb_fr),
         '{:.4f}'.format(id_fr),
         )
   
   ##########
   # Finish up ... 
   plt.legend(loc='upper left',framealpha=None,frameon=False)
   plt.tight_layout()
   plt.savefig(dir+'/roc.pdf')
   plt.clf()

   ##############
   # EFF CURVES #
   ##############

   # Binning
   bin_edges = np.linspace(0., 4., 8, endpoint=False)
   bin_edges = np.append( bin_edges, np.linspace(4., 8., 4, endpoint=False) )
   bin_edges = np.append( bin_edges, np.linspace(8., 10., 2, endpoint=True) )
   bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
   bin_widths = (bin_edges[1:] - bin_edges[:-1])
   bin_width = bin_widths[0]
   bin_widths /= bin_width
   #print("bin_edges",bin_edges)
   #print("bin_centres",bin_centres)
   #print("bin_widths",bin_widths)
   #print("bin_width",bin_width)

   plt.figure()
   ax = plt.subplot(111)

   has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_trk_ = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
   has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_pfgsf_ = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.5)
   if has_pfgsf_branches is False : has_pfgsf_ = has_gsf_ #@@ HACK
   curves = [{"label":"EGamma", "var":egamma.trk_pt,"mask":(egamma.is_e)&(has_trk_),"condition":(has_pfgsf_)},
             {"label":"Open",   "var":test.trk_pt,  "mask":(test.is_e)&(has_trk),   "condition":(has_gsf)},
             {"label":"VL seed","var":test.trk_pt,  "mask":(test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>unb_score[unb_VL])},
             {"label":"L seed", "var":test.trk_pt,  "mask":(test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>unb_score[unb_L])},
             {"label":"M seed", "var":test.trk_pt,  "mask":(test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>unb_score[unb_M])},
             {"label":"T seed", "var":test.trk_pt,  "mask":(test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>unb_score[unb_T])},
             {"label":"VT seed","var":test.trk_pt,  "mask":(test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>unb_score[unb_VT])},
             #{"label":"VL seed","var":test.trk_pt,  "mask":(test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>0.19)},
             #{"label":"L seed", "var":test.trk_pt,  "mask":(test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>1.20)},
             #{"label":"M seed", "var":test.trk_pt,  "mask":(test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>2.02)},
             #{"label":"T seed", "var":test.trk_pt,  "mask":(test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>3.05)},
             ]
             
   for idx,curve in enumerate(curves) :
      his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=bin_edges)
      his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=bin_edges)
      ax.errorbar(x=bin_centres,
                  y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ],
                  yerr=[ np.sqrt(x)/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ],
                  #color=None,
                  label='{:s} (mean={:5.3f})'.format(curve["label"],float(his_passed.sum())/float(his_total.sum())),
                  marker='.', elinewidth=1., capsize=1.) #linestyle='None', 
      
   ##########
   # Finish up ... 
   plt.title('Efficiency as a function of KF track pT')
   plt.xlabel('Transverse momentum (GeV)')
   plt.ylabel('Efficiency')
   ax.set_xlim(bin_edges[0],bin_edges[-1])
   plt.ylim([0., 1.])
   plt.legend(loc='best')
   plt.tight_layout()
   plt.savefig(dir+'/eff.pdf')
   plt.clf()

   #################
   # MISTAG CURVES #
   #################

   plt.figure()
   ax = plt.subplot(111)

   has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_trk_ = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
   has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_pfgsf_ = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.5)
   if has_pfgsf_branches is False : has_pfgsf_ = has_gsf_ #@@ HACK
   curves = [{"label":"EGamma", "var":egamma.trk_pt,"mask":(~egamma.is_e)&(has_trk_),"condition":(has_pfgsf_)},
             {"label":"Open",   "var":test.trk_pt,  "mask":(~test.is_e)&(has_trk),   "condition":(has_gsf)},
             {"label":"VL seed","var":test.trk_pt,  "mask":(~test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>unb_score[unb_VL])},
             {"label":"L seed", "var":test.trk_pt,  "mask":(~test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>unb_score[unb_L])},
             {"label":"M seed", "var":test.trk_pt,  "mask":(~test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>unb_score[unb_M])},
             {"label":"T seed", "var":test.trk_pt,  "mask":(~test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>unb_score[unb_T])},
             {"label":"VT seed","var":test.trk_pt,  "mask":(~test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>unb_score[unb_VT])},
             #{"label":"VL seed","var":test.trk_pt,  "mask":(~test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>0.19)},
             #{"label":"L seed", "var":test.trk_pt,  "mask":(~test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>1.20)},
             #{"label":"M seed", "var":test.trk_pt,  "mask":(~test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>2.02)},
             #{"label":"T seed", "var":test.trk_pt,  "mask":(~test.is_e)&(has_trk),   "condition":(has_gsf)&(test.gsf_bdtout1>3.05)},
             ]
             
   for idx,curve in enumerate(curves) :
      his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=bin_edges)
      his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=bin_edges)
      ax.errorbar(x=bin_centres,
                  y=[ x/y if y > 0. else 0. for x,y in zip(his_passed,his_total) ],
                  yerr=[ np.sqrt(x)/y if y > 0. else 0. for x,y in zip(his_passed,his_total) ],
                  #color=None,
                  label='{:s} (mean={:6.4f})'.format(curve["label"],float(his_passed.sum())/float(his_total.sum()) if his_total.sum() > 0 else 0.),
                  marker='.', elinewidth=1., capsize=1.) #linestyle='None', 
      
   ##########
   # Finish up ... 
   plt.title('Mistag rate as a function of KF track pT')
   plt.xlabel('Transverse momentum (GeV)')
   plt.ylabel('Mistag rate')
   plt.gca().set_yscale('log')
   ax.set_xlim(bin_edges[0],bin_edges[-1])
   ax.set_ylim([0.0001, 1.])
   plt.legend(loc='best')
   plt.tight_layout()
   plt.savefig(dir+'/mistag.pdf')
   plt.clf()
