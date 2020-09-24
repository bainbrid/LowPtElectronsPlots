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
def mauro(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### MAURO ##########################################################')

   #############
   # ROC CURVE #
   #############

   plt.figure(figsize=(6,6))
   ax = plt.subplot(111)
   plt.title('Efficiency and mistag rate w.r.t. GSF tracks')
   plt.xlim(1.e-3,1.1)
   plt.ylim([0., 0.6]) if AxE is True else plt.ylim([0., 1.03]) 
   plt.xlabel('FPR')
   plt.ylabel('TPR')
   ax.tick_params(axis='x', pad=10.)
   plt.gca().set_xscale('log')
   plt.grid(True)

   ########################################
   # "by chance" line
   plt.plot(np.arange(0.,1.,plt.xlim()[0]),np.arange(0.,1.,plt.xlim()[0]),'k--',lw=0.5)

   ########################################
   # Low-pT GSF electrons + ROC curves

   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_pfgsf = (test.has_pfgsf) & (test.pfgsf_pt>0.5) & (np.abs(test.pfgsf_eta)<2.5)
   #has_gsf |= has_pfgsf
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)

   # Eff and FR
   if AxE is True :
      denom = test.is_e; numer = has_ele&denom;
      ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   else :
      denom = has_gsf&test.is_e; numer = has_ele&denom;
      ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~test.is_e); numer = has_ele&denom;
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([ele_fr], [ele_eff],
            marker='o',color='blue', markersize=8, linestyle=None)

   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test['training_out'][has_ele])
   id_auc = roc_auc_score(test.is_e[has_ele],test['training_out'][has_ele])
   plt.plot(id_fpr*ele_fr, id_tpr*ele_eff,
            linestyle='solid', color='black', linewidth=1.0,
            label='Low-pT GSF electron + ID, AUC={:.3f}'.format(id_auc))
   
   # Unbiased seed BDT
   ele_unb_fpr,ele_unb_tpr,ele_unb_score = roc_curve(test.is_e[has_ele],test.gsf_bdtout1[has_ele])
   ele_unb_auc = roc_auc_score(test.is_e[has_ele],test.gsf_bdtout1[has_ele])
   plt.plot(ele_unb_fpr*ele_fr, ele_unb_tpr*ele_eff,
            linestyle='solid', color='blue', linewidth=1.0,
            label='Low-pT GSF electron + unbiased seed BDT, AUC={:.3f}'.format(ele_unb_auc))

   # Biased seed BDT
   ele_b_fpr,ele_b_tpr,ele_b_score = roc_curve(test.is_e[has_ele],test.gsf_bdtout2[has_ele])
   ele_b_auc = roc_auc_score(test.is_e[has_ele],test.gsf_bdtout2[has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
   plt.plot(ele_b_fpr*ele_fr, ele_b_tpr*ele_eff,
            linestyle='dashed', color='blue', linewidth=0.5,
            label='Low-pT GSF electron + biased seed BDT, AUC={:.3f}'.format(ele_b_auc))

   ########################################
   # Low-pT GSF tracks + ROC curves

   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_pfgsf = (test.has_pfgsf) & (test.pfgsf_pt>0.5) & (np.abs(test.pfgsf_eta)<2.5)
   #has_gsf |= has_pfgsf
   has_ele = None

   # Eff and FR
   if AxE is True :
      denom = test.is_e; numer = has_gsf&denom;
      gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   else :
      denom = has_gsf&test.is_e; numer = has_gsf&denom;
      gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~test.is_e); numer = has_gsf&denom;
   gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([gsf_fr], [gsf_eff],
            marker='o',color='red', markersize=8, linestyle='None')

   # Unbiased seed BDT
   gsf_unb_fpr,gsf_unb_tpr,gsf_unb_score = roc_curve(test.is_e[has_gsf],test.gsf_bdtout1[has_gsf])
   gsf_unb_auc = roc_auc_score(test.is_e[has_gsf],test.gsf_bdtout1[has_gsf])
   plt.plot(gsf_unb_fpr*gsf_fr, gsf_unb_tpr*gsf_eff,
            linestyle='solid', color='red', linewidth=1.0,
            label='Low-pT GSF track + unbiased seed BDT, AUC={:.3f}'.format(gsf_unb_auc))

   # Biased seed BDT
   gsf_b_fpr,gsf_b_tpr,gsf_b_score = roc_curve(test.is_e[has_gsf],test.gsf_bdtout2[has_gsf])
   gsf_b_auc = roc_auc_score(test.is_e[has_gsf],test.gsf_bdtout2[has_gsf]) if len(set(test.is_e[has_gsf])) > 1 else 0.
   plt.plot(gsf_b_fpr*gsf_fr, gsf_b_tpr*gsf_eff,
            linestyle='dashed', color='red', linewidth=0.5,
            label='Low-pT GSF track + biased seed BDT, AUC={:.3f}'.format(gsf_b_auc))
   
   ########################################
   # EGamma GSF tracks and PF GSF electrons

   has_gsf = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_pfgsf = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.5)
   #has_gsf |= has_pfgsf
   #has_gsf |= egamma.seed_ecal_driven
   has_ele = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)

   # Eff and FR (EGamma GSF tracks)
   if AxE is True :
      denom = egamma.is_e; numer = has_pfgsf&denom
      eg_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   else :
      denom = has_gsf&egamma.is_e; numer = has_pfgsf&denom
      eg_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~egamma.is_e); numer = has_pfgsf&denom
   eg_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([eg_fr], [eg_eff],
            marker='o', color='green', markersize=8, linestyle='None',
            label='EGamma GSF track')

   # Eff and FR (EGamma PF GSF electrons)
   if AxE is True :
      denom = egamma.is_e; numer = has_ele&denom
      pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   else :
      denom = has_gsf&egamma.is_e; numer = has_ele&denom
      pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~egamma.is_e); numer = has_ele&denom
   pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([pf_fr], [pf_eff],
            marker='o', color='purple', markersize=8, linestyle='None',
            label='PF GSF electron')

   print('eff: {:.3f}, mistag: {:.4f}'.format(ele_eff,ele_fr),'Low-pT GSF electrons + ID')
   print('eff: {:.3f}, mistag: {:.4f}'.format(gsf_eff,gsf_fr),'Low-pT GSF electrons + unbiased seed BDT')
   print('eff: {:.3f}, mistag: {:.4f}'.format(eg_eff,eg_fr),'EGamma GSF tracks')
   print('eff: {:.3f}, mistag: {:.4f}'.format(pf_eff,pf_fr),'EGamma GSF electrons')
   
   ##########
   # EGamma GSF electrons (ECAL-driven)

#   has_gsf |= egamma.seed_ecal_driven
#   has_ele &= egamma.seed_ecal_driven
#
#   if AxE is True :
#      denom = egamma.is_e; numer = has_ele&denom
#      _eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   else :
#      denom = has_gsf&egamma.is_e; numer = has_ele&denom
#      _eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   denom = has_gsf&(~egamma.is_e); numer = has_ele&denom
#   _fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   plt.plot([_fr], [_eff],
#            marker='o', markerfacecolor='none', markeredgecolor='purple', markersize=8, linestyle='None',
#            label='PF GSF electron (ECAL-driven)')
#   print('eff: {:.3f}, mistag: {:.4f}'.format(_eff,_fr),'EGamma GSF electrons (ECAL-driven)')

   # "New" WPs 
   unb_L   = np.abs(gsf_unb_fpr*gsf_fr-eg_fr*10.).argmin()
   unb_M   = np.abs(gsf_unb_fpr*gsf_fr-eg_fr*3.).argmin()
   unb_T   = np.abs(gsf_unb_fpr*gsf_fr-eg_fr).argmin()
   unb_VT  = np.abs(gsf_unb_tpr*gsf_eff-eg_eff).argmin()
   unb_ELE = np.abs(gsf_unb_fpr*gsf_fr-pf_fr).argmin() # same FR
   id_ELE = np.abs(id_fpr*ele_fr-pf_fr).argmin() # same FR

   print("EG GSF track:      FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(eg_fr,eg_eff,np.nan))
   print("VLoose   (10% FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(gsf_fr, gsf_eff,np.nan))
   print("Loose    (x10 FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(gsf_unb_fpr[unb_L]*gsf_fr,  gsf_unb_tpr[unb_L]*gsf_eff,  gsf_unb_score[unb_L]))
   print("Medium    (x3 FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(gsf_unb_fpr[unb_M]*gsf_fr,  gsf_unb_tpr[unb_M]*gsf_eff,  gsf_unb_score[unb_M]))
   print("Tight   (same FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(gsf_unb_fpr[unb_T]*gsf_fr,  gsf_unb_tpr[unb_T]*gsf_eff,  gsf_unb_score[unb_T]))
   print("VTight (same eff): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(gsf_unb_fpr[unb_VT]*gsf_fr, gsf_unb_tpr[unb_VT]*gsf_eff, gsf_unb_score[unb_VT]))
   print("PF GSF electron:   FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(pf_fr,pf_eff,np.nan))
   print("Unb/PF  (same FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(ele_unb_fpr[unb_ELE]*ele_fr, ele_unb_tpr[unb_ELE]*ele_eff, ele_unb_score[unb_ELE]))
   print("ID/PF   (same FR): FR, Eff, score:","{:.4f}, {:.3f}, {:5.2f} ".format(id_fpr[id_ELE]*ele_fr, id_tpr[id_ELE]*ele_eff, id_score[id_ELE]))

   x,y = gsf_unb_fpr[unb_L]*gsf_fr,gsf_unb_tpr[unb_L]*gsf_eff
   plt.plot([x], [y],  marker='o', markerfacecolor='none', markeredgecolor='green', markersize=4)
   plt.text(x, y-0.02, "L", fontsize=10, ha='center', va='center', color='green' )
   
   x,y = gsf_unb_fpr[unb_M]*gsf_fr,gsf_unb_tpr[unb_M]*gsf_eff
   plt.plot([x], [y],  marker='o', markerfacecolor='none', markeredgecolor='green', markersize=4)
   plt.text(x, y-0.02, "M", fontsize=10, ha='center', va='center', color='green' )
   
   x,y = gsf_unb_fpr[unb_T]*gsf_fr,gsf_unb_tpr[unb_T]*gsf_eff
   plt.plot([x], [y],  marker='o', markerfacecolor='none', markeredgecolor='green', markersize=4)
   plt.text(x, y-0.02, "T", fontsize=10, ha='center', va='center', color='green' )
   
   x,y = gsf_unb_fpr[unb_VT]*gsf_fr,gsf_unb_tpr[unb_VT]*gsf_eff
   plt.plot([x], [y], marker='o', markerfacecolor='none', markeredgecolor='green', markersize=4)
   plt.text(x, y-0.02, "VT", fontsize=10, ha='center', va='center', color='green' )
   
   x,y = ele_unb_fpr[unb_ELE]*ele_fr,ele_unb_tpr[unb_ELE]*ele_eff
   plt.plot([x], [y], marker='o', markerfacecolor='none', markeredgecolor='purple', markersize=4)
   plt.text(x, y-0.02, "E", fontsize=10, ha='center', va='center', color='purple' )

   x,y = id_fpr[id_ELE]*ele_fr,id_tpr[id_ELE]*ele_eff
   plt.plot([x], [y], marker='o', markerfacecolor='none', markeredgecolor='purple', markersize=4)
   plt.text(x, y-0.02, "E", fontsize=10, ha='center', va='center', color='purple' )

   # Original WPs
   # https://github.com/cms-sw/cmssw/blob/CMSSW_10_2_X/RecoEgamma/EgammaElectronProducers/python/lowPtGsfElectronSeeds_cfi.py
   #orig_VL  = np.abs(gsf_unb_score-0.19).argmin()
   #orig_L   = np.abs(gsf_unb_score-1.20).argmin()
   #orig_M   = np.abs(gsf_unb_score-2.02).argmin()
   #orig_T   = np.abs(gsf_unb_score-3.05).argmin()
   #orig_ELE = np.abs(gsf_unb_score-5.26).argmin() # same FR?

   #x,y = gsf_unb_fpr[orig_VL]*gsf_fr,gsf_unb_tpr[orig_VL]*gsf_eff
   #plt.plot([x],[y], marker='^', markerfacecolor='none', markeredgecolor='green', markersize=4)
   #plt.text(x, y-0.02, "VL", fontsize=10, ha='center', va='center', color='green' )

   #x,y = gsf_unb_fpr[orig_L]*gsf_fr,gsf_unb_tpr[orig_L]*gsf_eff
   #plt.plot([x],[y], marker='^', markerfacecolor='none', markeredgecolor='green', markersize=4)
   #plt.text(x, y-0.02, "L", fontsize=10, ha='center', va='center', color='green' )
   
   #x,y = gsf_unb_fpr[orig_M]*gsf_fr,gsf_unb_tpr[orig_M]*gsf_eff
   #plt.plot([x],[y], marker='^', markerfacecolor='none', markeredgecolor='green', markersize=4)
   #plt.text(x, y-0.02, "M", fontsize=10, ha='center', va='center', color='green' )
   
   #x,y = gsf_unb_fpr[orig_T]*gsf_fr,gsf_unb_tpr[orig_T]*gsf_eff
   #plt.plot([x],[y], marker='^', markerfacecolor='none', markeredgecolor='green', markersize=4)
   #plt.text(x, y-0.02, "T", fontsize=10, ha='center', va='center', color='green' )

   #x,y = gsf_unb_fpr[orig_ELE]*gsf_fr,gsf_unb_tpr[orig_ELE]*gsf_eff
   #plt.plot([x],[y], marker='^', markerfacecolor='none', markeredgecolor='purple', markersize=4)
   #plt.text(x, y+0.02, "E", fontsize=10, ha='center', va='center', color='purple' )
   
   ##########
   # Finish up ... 
   plt.legend(loc='upper left',framealpha=None,frameon=False)
   plt.tight_layout()
   plt.savefig(dir+'/roc.pdf')
   plt.clf()
   plt.close()

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
   has_pfgsf = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.5)
   has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
   curves = [
      {"label":"EG GSF","var":egamma.gsf_pt,"mask":(egamma.is_e)&(has_gsf_),"condition":(has_pfgsf),"colour":"green","size":7,},
      {"label":"PF ELE","var":egamma.gsf_pt,"mask":(egamma.is_e)&(has_gsf_),"condition":(has_ele_),"colour":"purple","size":7,},
      {"label":"Unbiased","var":test.gsf_pt, "mask":(test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_ELE]),"colour":"blue","size":7,},
      {"label":"ID","var":test.gsf_pt, "mask":(test.is_e)&(has_gsf), "condition":(has_ele)&(test['training_out']>id_score[id_ELE]),"colour":"black","size":7,},
      #{"label":"Track (VT)","var":test.gsf_pt, "mask":(test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_VT]),"colour":"red","size":7,},
      #{"label":"Track (T)", "var":test.gsf_pt, "mask":(test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_T]),"colour":"red","size":6,},
      #{"label":"Track (M)", "var":test.gsf_pt, "mask":(test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_M]),"colour":"red","size":5,},
      #{"label":"Track (L)", "var":test.gsf_pt, "mask":(test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_L]),"colour":"red","size":4,},
      #{"label":"Track (VL)","var":test.gsf_pt, "mask":(test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_VL]),"colour":"red","size":3,},
      {"label":"Open",     "var":test.gsf_pt, "mask":(test.is_e)&(has_gsf), "condition":(has_ele),"colour":"red","size":7,},
      ]
             
   for idx,curve in enumerate(curves) :
      #print("label:",curve["label"])
      his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=bin_edges)
      his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=bin_edges)
      x=bin_edges[:-1]
      y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ]
      yhigh=[ binomial_hpdr(p,t)[1]-(p/t) if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
      ylow =[ (p/t)-binomial_hpdr(p,t)[0] if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
      yerr =[ylow,yhigh]
      label='{:s} (mean={:5.3f})'.format(curve["label"],
                                         float(his_passed.sum())/float(his_total.sum()) \
                                            if his_total.sum() > 0 else 0.)
      ax.errorbar(x=x,
                  y=y,
                  yerr=yerr,
                  #color=None,
                  label=label,
                  marker='o',
                  color=curve["colour"],
                  markerfacecolor="white",
                  markersize=curve["size"],
                  linewidth=0.5,
                  elinewidth=0.5)

   ##########
   # Finish up ... 
   plt.title('Efficiency as a function of GSF track pT')
   plt.xlabel('Transverse momentum (GeV)')
   plt.ylabel('Efficiency')
   ax.set_xlim(bin_edges[0],bin_edges[-2])
   plt.ylim([0., 1.])
   plt.legend(loc='best')
   plt.tight_layout()
   plt.savefig(dir+'/eff.pdf')
   plt.clf()
   plt.close()

   #################
   # MISTAG CURVES #
   #################

   plt.figure()
   ax = plt.subplot(111)

   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
   has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_pfgsf = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.5)
   has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
   curves = [
      {"label":"EG GSF","var":egamma.gsf_pt,"mask":(~egamma.is_e)&(has_gsf_),"condition":(has_pfgsf),"colour":"green","size":7,},
      {"label":"PF ELE","var":egamma.gsf_pt,"mask":(~egamma.is_e)&(has_gsf_),"condition":(has_ele_),"colour":"purple","size":7,},
      {"label":"Unbiased","var":test.gsf_pt, "mask":(~test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_ELE]),"colour":"blue","size":7,},
      {"label":"ID","var":test.gsf_pt, "mask":(~test.is_e)&(has_gsf), "condition":(has_ele)&(test['training_out']>id_score[id_ELE]),"colour":"black","size":7,},
      #{"label":"Track (VT)","var":test.gsf_pt, "mask":(~test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_VT]),"colour":"red","size":7,},
      #{"label":"Track (T)", "var":test.gsf_pt, "mask":(~test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_T]),"colour":"red","size":6,},
      #{"label":"Track (M)", "var":test.gsf_pt, "mask":(~test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_M]),"colour":"red","size":5,},
      #{"label":"Track (L)", "var":test.gsf_pt, "mask":(~test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_L]),"colour":"red","size":4,},
      #{"label":"Track (VL)","var":test.gsf_pt, "mask":(~test.is_e)&(has_gsf), "condition":(has_ele)&(test.gsf_bdtout1>gsf_unb_score[unb_VL]),"colour":"red","size":3,},
      {"label":"Open",     "var":test.gsf_pt, "mask":(~test.is_e)&(has_gsf), "condition":(has_ele),"colour":"red","size":7,},
      ]
   
   for idx,curve in enumerate(curves) :
      #print("label:",curve["label"])
      his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=bin_edges)
      his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=bin_edges)
      x=bin_edges[:-1]
      y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ]
      yhigh=[ binomial_hpdr(p,t)[1]-(p/t) if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
      ylow =[ (p/t)-binomial_hpdr(p,t)[0] if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
      yerr =[ylow,yhigh]
      label='{:s} (mean={:5.3f})'.format(curve["label"],
                                         float(his_passed.sum())/float(his_total.sum()) \
                                            if his_total.sum() > 0 else 0.)
      ax.errorbar(x=x,
                  y=y,
                  yerr=yerr,
                  #color=None,
                  label=label,
                  marker='o',
                  color=curve["colour"],
                  markerfacecolor="white",
                  markersize=curve["size"],
                  linewidth=0.5,
                  elinewidth=0.5)
      
   ##########
   # Finish up ... 
   plt.title('Mistag rate as a function of GSF track pT')
   plt.xlabel('Transverse momentum (GeV)')
   plt.ylabel('Mistag rate')
   plt.gca().set_yscale('log')
   ax.set_xlim(bin_edges[0],bin_edges[-2])
   ax.set_ylim([0.0001, 1.])
   plt.legend(loc='best')
   plt.tight_layout()
   plt.savefig(dir+'/mistag.pdf')
   plt.clf()
   plt.close()
