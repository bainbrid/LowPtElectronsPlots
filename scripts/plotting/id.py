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
def id(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### ID #############################################################')

   #############
   # ROC CURVE #
   #############

   plt.figure()
   ax = plt.subplot(111)
   plt.title('Efficiency and mistag rate w.r.t. GSF tracks')
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
   # Low-pT GSF electrons + ID

   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_pfgsf = (test.has_pfgsf) & (test.pfgsf_pt>0.5) & (np.abs(test.pfgsf_eta)<2.5)
   #has_gsf |= has_pfgsf
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)

   # Eff and FR
   if AxE is True :
      ele_eff = float((has_ele&test.is_e).sum()) / float((test.is_e).sum()) \
          if float((test.is_e).sum()) > 0. else 0.
   else :
      ele_eff = float((has_ele&has_gsf&test.is_e).sum()) / float((has_gsf&test.is_e).sum()) \
          if float((has_gsf&test.is_e).sum()) > 0. else 0.
   ele_if = float((has_ele&has_gsf&(~test.is_e)).sum()) / float((has_gsf&(~test.is_e)).sum()) \
       if float((has_gsf&(~test.is_e)).sum()) > 0. else 0.

   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test['training_out'][has_ele])
   id_auc = roc_auc_score(test.is_e[has_ele],test['training_out'][has_ele])
   plt.plot(id_fpr*ele_if, id_tpr*ele_eff,
            linestyle='solid', color='blue', linewidth=1.,
            label='Low-pT GSF electron + ID, AUC={:.3f}'.format(id_auc))
   plt.plot([ele_if], [ele_eff],
            marker='o',color='blue', markersize=8, linestyle=None)

   ##########
   # Low-pT GSF tracks + unbiased seed BDT 
   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_pfgsf = (test.has_pfgsf) & (test.pfgsf_pt>0.5) & (np.abs(test.pfgsf_eta)<2.5)
   #has_gsf |= has_pfgsf

   if AxE is True :
      unb_eff = float((has_gsf&test.is_e).sum()) / float((test.is_e).sum()) \
          if float((test.is_e).sum()) > 0. else 0.
   else :
      unb_eff = float((has_gsf&test.is_e).sum()) / float((has_gsf&test.is_e).sum()) \
          if float((has_gsf&test.is_e).sum()) > 0. else 0.
   unb_fr = float((has_gsf&(~test.is_e)).sum()) / float((has_gsf&(~test.is_e)).sum()) \
       if float((has_gsf&(~test.is_e)).sum()) > 0. else 0.
   unb_fpr,unb_tpr,unb_score = roc_curve(test.is_e[has_gsf],test.gsf_bdtout1[has_gsf])
   unb_auc = roc_auc_score(test.is_e[has_gsf],test.gsf_bdtout1[has_gsf])
   plt.plot(unb_fpr*unb_fr, unb_tpr*unb_eff,
            linestyle='solid', color='red', linewidth=1.,
            label='Low-pT GSF electron + unbiased seed BDT, AUC={:.3f}'.format(unb_auc))
   plt.plot([unb_fr], [unb_eff],
            marker='o',color='red', markersize=8, linestyle='None')
   
   ##########
   # EGamma GSF tracks and PF GSF electrons
   has_gsf = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_pfgsf = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.5)
   #has_gsf |= has_pfgsf
   if has_pfgsf_branches is False : has_pfgsf = has_gsf #@@ HACK
   has_ele = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)

   if AxE is True :
      eg_eff = float((has_pfgsf&egamma.is_e).sum()) / float((egamma.is_e).sum()) \
          if float((egamma.is_e).sum()) > 0. else 0.
   else :
      eg_eff = float((has_pfgsf&has_gsf&egamma.is_e).sum()) / float((has_gsf&egamma.is_e).sum()) \
          if float((has_gsf&egamma.is_e).sum()) > 0. else 0.
   eg_fr = float((has_pfgsf&has_gsf&(~egamma.is_e)).sum()) / float((has_gsf&(~egamma.is_e)).sum()) \
       if float((has_gsf&(~egamma.is_e)).sum()) > 0. else 0.
   plt.plot([eg_fr], [eg_eff],
            marker='o', color='green', markersize=8, linestyle='None',
            label='EGamma GSF track')

   if AxE is True :
      pf_eff = float((has_ele&egamma.is_e).sum()) / float((egamma.is_e).sum()) \
          if float((egamma.is_e).sum()) > 0. else 0.
   else :
      pf_eff = float((has_ele&has_gsf&egamma.is_e).sum()) / float((has_gsf&egamma.is_e).sum()) \
          if float((has_gsf&egamma.is_e).sum()) > 0. else 0.
   pf_fr = float((has_ele&has_gsf&(~egamma.is_e)).sum()) / float((has_gsf&(~egamma.is_e)).sum()) \
       if float((has_gsf&(~egamma.is_e)).sum()) > 0. else 0.
   plt.plot([pf_fr], [pf_eff],
            marker='o', color='purple', markersize=8, linestyle='None',
            label='PF GSF electron')

   print('eff: {:.3f}, mistag: {:.4f}'.format(ele_eff,ele_if),'Low-pT GSF electrons + ID')
   print('eff: {:.3f}, mistag: {:.4f}'.format(unb_eff,unb_fr),'Low-pT GSF electrons + unbiased seed BDT')
   print('eff: {:.3f}, mistag: {:.4f}'.format(eg_eff,eg_fr),'EGamma GSF tracks')
   print('eff: {:.3f}, mistag: {:.4f}'.format(pf_eff,pf_fr),'EGamma GSF electrons')
   
   ##########
   # EGamma GSF electrons (ECAL-driven)
   has_gsf |= egamma.seed_ecal_driven
   has_ele &= egamma.seed_ecal_driven
   if AxE is True :
      _eff = float((has_ele&egamma.is_e).sum()) / float((egamma.is_e).sum()) \
          if float((egamma.is_e).sum()) > 0. else 0.
   else :
      _eff = float((has_ele&has_gsf&egamma.is_e).sum()) / float((has_gsf&egamma.is_e).sum()) \
          if float((has_gsf&egamma.is_e).sum()) > 0. else 0.
   _fr = float((has_ele&has_gsf&(~egamma.is_e)).sum()) / float((has_gsf&(~egamma.is_e)).sum()) \
       if float((has_gsf&(~egamma.is_e)).sum()) > 0. else 0.
   plt.plot([_fr], [_eff],
            marker='o', markerfacecolor='none', markeredgecolor='purple', markersize=8, linestyle='None',
            label='PF GSF electron (ECAL-driven)')
   print('eff: {:.3f}, mistag: {:.4f}'.format(_eff,_fr),'EGamma GSF electrons (ECAL-driven)')

   ##########
   # Working points, "newly tuned"
   id_VL = np.abs(id_fpr*ele_if-0.1).argmin()
   id_L  = np.abs(id_fpr*ele_if-pf_fr*10.).argmin()
   id_M  = np.abs(id_fpr*ele_if-pf_fr*3.).argmin()
   id_T  = np.abs(id_fpr*ele_if-pf_fr).argmin()
   id_VT = np.abs(id_tpr*ele_eff-pf_eff).argmin()
   print("EGamma GSF track:  FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(pf_fr,pf_eff,np.nan))
   print("VLoose   (10% FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(id_fpr[id_VL]*ele_if, id_tpr[id_VL]*ele_eff, id_score[id_VL]))
   print("Loose    (x10 FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(id_fpr[id_L]*ele_if,  id_tpr[id_L]*ele_eff,  id_score[id_L]))
   print("Medium    (x3 FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(id_fpr[id_M]*ele_if,  id_tpr[id_M]*ele_eff,  id_score[id_M]))
   print("Tight   (same FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(id_fpr[id_T]*ele_if,  id_tpr[id_T]*ele_eff,  id_score[id_T]))
   print("VTight (same eff): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(id_fpr[id_VT]*ele_if, id_tpr[id_VT]*ele_eff, id_score[id_VT]))
   plt.plot([id_fpr[id_VL]*ele_if], [id_tpr[id_VL]*ele_eff], marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([id_fpr[id_L]*ele_if],  [id_tpr[id_L]*ele_eff],  marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([id_fpr[id_M]*ele_if],  [id_tpr[id_M]*ele_eff],  marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([id_fpr[id_T]*ele_if],  [id_tpr[id_T]*ele_eff],  marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([id_fpr[id_VT]*ele_if], [id_tpr[id_VT]*ele_eff], marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4)
   # Original ID WPs
   ele_T  = np.abs(id_score-4.24).argmin()
   ele_VT  = np.abs(id_score-4.93).argmin()
   plt.plot([id_fpr[ele_T]*ele_if],  [id_tpr[ele_T]*ele_eff],  marker='^', markerfacecolor='none', markeredgecolor='k', markersize=4)
   plt.plot([id_fpr[ele_VT]*ele_if],  [id_tpr[ele_VT]*ele_eff],  marker='^', markerfacecolor='none', markeredgecolor='k', markersize=4)
   
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

   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
   has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
   curves = [{"label":"EGamma", "var":egamma.trk_pt,"mask":(egamma.is_e)&(has_gsf_),"condition":(has_ele_)},
             {"label":"Open",   "var":test.gsf_pt,  "mask":(test.is_e)&(has_gsf),   "condition":(has_ele)},
             {"label":"VL seed","var":test.gsf_pt,  "mask":(test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>id_score[id_VL])},
             {"label":"L seed", "var":test.gsf_pt,  "mask":(test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>id_score[id_L])},
             {"label":"M seed", "var":test.gsf_pt,  "mask":(test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>id_score[id_M])},
             {"label":"T seed", "var":test.gsf_pt,  "mask":(test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>id_score[id_T])},
             {"label":"VT seed","var":test.gsf_pt,  "mask":(test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>id_score[id_VT])},
             #{"label":"T seed", "var":test.gsf_pt,  "mask":(test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>4.24)},
             #{"label":"VT seed", "var":test.gsf_pt,  "mask":(test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>4.93)},
             ]
             
   for idx,curve in enumerate(curves) :
      #print("label:",curve["label"])
      his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=bin_edges)
      his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=bin_edges)
      ax.errorbar(x=bin_centres,
                  y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ],
                  yerr=[ np.sqrt(x)/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ],
                  #color=None,
                  label='{:s} (mean={:5.3f})'.format(curve["label"],
                                                     float(his_passed.sum())/float(his_total.sum()) \
                                                        if his_total.sum() > 0 else 0.),
                  marker='.', elinewidth=1., capsize=1.) #linestyle='None', 
      
   ##########
   # Finish up ... 
   plt.title('Efficiency as a function of GSF track pT')
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

   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
   has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
   curves = [{"label":"EGamma", "var":egamma.trk_pt,"mask":(~egamma.is_e)&(has_gsf_),"condition":(has_ele_)},
             {"label":"Open",   "var":test.gsf_pt,  "mask":(~test.is_e)&(has_gsf),   "condition":(has_ele)},
             {"label":"VL seed","var":test.gsf_pt,  "mask":(~test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>id_score[id_VL])},
             {"label":"L seed", "var":test.gsf_pt,  "mask":(~test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>id_score[id_L])},
             {"label":"M seed", "var":test.gsf_pt,  "mask":(~test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>id_score[id_M])},
             {"label":"T seed", "var":test.gsf_pt,  "mask":(~test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>id_score[id_T])},
             {"label":"VT seed","var":test.gsf_pt,  "mask":(~test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>id_score[id_VT])},
             #{"label":"T seed", "var":test.gsf_pt,  "mask":(~test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>4.24)},
             #{"label":"VT seed", "var":test.gsf_pt,  "mask":(~test.is_e)&(has_gsf),   "condition":(has_ele)&(test['training_out']>4.93)},
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
   plt.title('Mistag rate as a function of GSF track pT')
   plt.xlabel('Transverse momentum (GeV)')
   plt.ylabel('Mistag rate')
   plt.gca().set_yscale('log')
   ax.set_xlim(bin_edges[0],bin_edges[-1])
   ax.set_ylim([0.0001, 1.])
   plt.legend(loc='best')
   plt.tight_layout()
   plt.savefig(dir+'/mistag.pdf')
   plt.clf()
