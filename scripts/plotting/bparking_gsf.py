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
def bparking_gsf(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### BPARKING GSF ##########################################################')

   #############
   # ROC CURVE #
   #############

   plt.figure(figsize=(6,6))
   ax = plt.subplot(111)
   plt.title('Low-pT electron performance (BParking)')
   plt.xlim(1.e-3,1.)
   plt.ylim([0., 1.])
   plt.xlabel('Mistag rate (w.r.t. GSF tracks, pT > 0.5 GeV)')
   plt.ylabel('Efficiency (w.r.t. GSF tracks, pT > 0.5 GeV)')
   ax.tick_params(axis='x', pad=10.)
   plt.gca().set_xscale('log')
   plt.grid(True)

   ########################################
   # "by chance" line

   plt.plot(np.arange(0.,1.,plt.xlim()[0]),np.arange(0.,1.,plt.xlim()[0]),ls='dotted',lw=0.5,label="By chance")

   ########################################
   # GSF track (pT > 0.5 GeV, VL WP for Seed BDT)

   plt.plot([1.], [1.],
            marker='o', markerfacecolor='red', markeredgecolor='red', 
            markersize=8,linestyle='none',
            label='Low-pT GSF track',
            )

#   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
#   unb_branch = 'gsf_bdtout1'
#   unb_fpr,unb_tpr,unb_score = roc_curve(test.is_e[has_gsf],test[unb_branch][has_gsf])
#   unb_auc = roc_auc_score(test.is_e[has_gsf],test[unb_branch][has_gsf]) if len(set(test.is_e[has_gsf])) > 1 else 0.
#   plt.plot(unb_fpr, 
#            unb_tpr,
#            linestyle='solid', color='red', linewidth=1.0,
#            label='Seeding, unbiased (AUC={:.3f})'.format(unb_auc))

   ########################################
   # Electron (pT > 0.5 GeV, VL WP for Seed BDT)

   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
   denom = has_gsf&test.is_e; numer = has_ele&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~test.is_e); numer = has_ele&denom;
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([ele_fr], [ele_eff],
            marker='o', markerfacecolor='blue', markeredgecolor='blue', 
            markersize=8,linestyle='none',
            label='Low-pT electron',
            )

   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test['ele_mva_value_depth15'][has_ele])
   id_auc = roc_auc_score(test.is_e[has_ele],test['ele_mva_value_depth15'][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
   plt.plot(id_fpr*ele_fr, 
           id_tpr*ele_eff,
            linestyle='solid', color='blue', linewidth=1.0,
            label='2020Feb24 (AUC={:.3f})'.format(id_auc))
   
   id2_fpr,id2_tpr,id2_score = roc_curve(test.is_e[has_ele],test['training_out'][has_ele]) # replace with ele_mva_value
   id2_auc = roc_auc_score(test.is_e[has_ele],test['training_out'][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0. # replace with ele_mva_value
   plt.plot(id2_fpr*ele_fr, 
            id2_tpr*ele_eff,
            linestyle='dashed', color='blue', linewidth=1.0,
            label='2019Aug07 (AUC={:.3f})'.format(id2_auc))

   ########################################
   # EGamma PF GSF electrons

#   has_gsf = (egamma.has_gsf) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
#   has_pfgsf = (egamma.has_pfgsf) & (egamma.pfgsf_pt>0.5) & (np.abs(egamma.pfgsf_eta)<2.5)
#   denom = has_gsf&egamma.is_e; numer = has_pfgsf&denom
#   eg_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   denom = has_gsf&(~egamma.is_e); numer = has_pfgsf&denom
#   eg_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   plt.plot([eg_fr], [eg_eff],
#            marker='o', color='green', 
#            markersize=8, linestyle='none',
#            label='(Reduced) EGamma GSF track')

   has_gsf = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
   has_ele = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
   denom = has_gsf&egamma.is_e; numer = has_ele&denom
   pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_gsf&(~egamma.is_e); numer = has_ele&denom
   pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([pf_fr], [pf_eff],
            marker='o', color='purple', 
            markersize=8, linestyle='none',
            label='PF electron')

   pf_id_fpr,pf_id_tpr,pf_id_score = roc_curve(egamma.is_e[has_ele],egamma['ele_mva_value_retrained'][has_ele])
   pf_id_auc = roc_auc_score(egamma.is_e[has_ele],egamma['ele_mva_value_retrained'][has_ele]) if len(set(egamma.is_e[has_ele])) > 1 else 0.
   plt.plot(pf_id_fpr*pf_fr, 
            pf_id_tpr*pf_eff,
            linestyle='solid', color='purple', linewidth=1.0,
            label='ID, retrain (AUC={:.3f})'.format(pf_id_auc))

   pf_id2_fpr,pf_id2_tpr,pf_id2_score = roc_curve(egamma.is_e[has_ele],egamma['ele_mva_value'][has_ele])
   pf_id2_auc = roc_auc_score(egamma.is_e[has_ele],egamma['ele_mva_value'][has_ele]) if len(set(egamma.is_e[has_ele])) > 1 else 0.
   plt.plot(pf_id2_fpr*pf_fr, 
            pf_id2_tpr*pf_eff,
            linestyle='dashed', color='purple', linewidth=1.0,
            label='ID, EGamma (AUC={:.3f})'.format(pf_id2_auc))

   #################
   # Working points

   id_ELE = np.abs(id_tpr*ele_eff-pf_eff).argmin()
   same_eff = test['ele_mva_value_depth15']>id_score[id_ELE] # training_out
   
   x,y = id_fpr[id_ELE]*ele_fr,id_tpr[id_ELE]*ele_eff
   #plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='black', markersize=8)
   #plt.text(x, y+0.03, "Tight", fontsize=8, ha='center', va='center', color='black' )

   id_ELE = np.abs(id_fpr*ele_fr-pf_fr).argmin()
   same_fr = test['ele_mva_value_depth15']>id_score[id_ELE] # training_out

   x,y = id_fpr[id_ELE]*ele_fr,id_tpr[id_ELE]*ele_eff
   plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='blue', markersize=8)
   #plt.text(x, y+0.03, "Loose", fontsize=8, ha='center', va='center', color='black' )

   ##########
   # Finish up ... 
   plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
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
   bin_edges = np.append( bin_edges, np.linspace(8., 12., 3, endpoint=True) )
   bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
   bin_widths = (bin_edges[1:] - bin_edges[:-1])
   bin_width = bin_widths[0]
   bin_widths /= bin_width
   #print("bin_edges",bin_edges)
   #print("bin_centres",bin_centres)
   #print("bin_widths",bin_widths)
   #print("bin_width",bin_width)

   tuple = ([
      'gen_pt',
      'gsf_pt',
      'gsf_mode_pt',
      'gsf_dxy',
      'gsf_dz',
      'rho',
      ],
   [
      bin_edges,
      bin_edges,
      bin_edges,
      np.linspace(0.,3.3,12),
      np.linspace(0.,22.,12),
      np.linspace(0.,44.,12),
      ],
   [
      'Generator-level transverse momentum (GeV)',
      'Transverse momentum (GeV)',
      'Mode transverse momentum (GeV)',
      'Transverse impact parameter w.r.t. beamspot (cm)',
      'Longitudinal impact parameter w.r.t. beamspot (cm)',
      'Median energy density from UE/pileup (GeV / unit area)',
      ])

   print("Efficiency curves ...")
   for attr,binning,xlabel in zip(*tuple) :
      print(attr)

      plt.figure()
      ax = plt.subplot(111)

      has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
      has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
      has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
      has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
      curves = [
         {"label":"Low-pT electron","var":test[attr],"mask":(test.is_e)&(has_gsf),"condition":(has_ele),"colour":"blue","fill":True,"size":8,},
         {"label":"Same mistag rate","var":test[attr],"mask":(test.is_e)&(has_gsf),"condition":(has_ele)&(same_fr),"colour":"blue","fill":False,"size":8,},
         {"label":"PF electron","var":egamma[attr],"mask":(egamma.is_e)&(has_gsf_),"condition":(has_ele_),"colour":"purple","fill":True,"size":8,},
         ]
             
      for idx,curve in enumerate(curves) :
         # print("label:",curve["label"])
         his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=binning)
         his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=binning)
         x=binning[:-1]
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
                     markerfacecolor = curve["colour"] if curve["fill"] else "white",
                     markersize=curve["size"],
                     linewidth=0.5,
                     elinewidth=0.5)
         
      # #########
      # Finish up ... 
      plt.title('Low-pT electron performance (BParking)')
      plt.xlabel(xlabel)
      plt.ylabel('Efficiency (w.r.t. GSF tracks, pT > 0.5 GeV)')
      ax.set_xlim(binning[0],binning[-2])
      plt.ylim([0., 1.])
      plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
      plt.tight_layout()
      plt.savefig(dir+'/eff_vs_{:s}.pdf'.format(attr))
      plt.clf()
      plt.close()

   #################
   # MISTAG CURVES #
   #################

   print("Mistag curves ...")
   for attr,binning,xlabel in zip(*tuple) :
      print(attr)

      plt.figure()
      ax = plt.subplot(111)

      has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
      has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
      has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
      has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
      curves = [
         {"label":"Low-pT electron","var":test[attr],"mask":(~test.is_e)&(has_gsf),"condition":(has_ele),"colour":"blue","fill":True,"size":8,},
         {"label":"Same mistag rate","var":test[attr],"mask":(~test.is_e)&(has_gsf),"condition":(has_ele)&(same_fr),"colour":"blue","fill":False,"size":8,},
         {"label":"PF electron","var":egamma[attr],"mask":(~egamma.is_e)&(has_gsf_),"condition":(has_ele_),"colour":"purple","fill":True,"size":8,},
         ]
   
      for idx,curve in enumerate(curves) :
         his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=binning)
         his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=binning)
         x=binning[:-1]
         y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ]
         yhigh=[ binomial_hpdr(p,t)[1]-(p/t) if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
         ylow =[ (p/t)-binomial_hpdr(p,t)[0] if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
         yerr =[ylow,yhigh]
         label='{:s} (mean={:6.4f})'.format(curve["label"],
                                            float(his_passed.sum())/float(his_total.sum()) \
                                               if his_total.sum() > 0 else 0.)
         ax.errorbar(x=x,
                     y=y,
                     yerr=yerr,
                     #color=None,
                     label=label,
                     marker='o',
                     color=curve["colour"],
                     markerfacecolor = curve["colour"] if curve["fill"] else "white",
                     markersize=curve["size"],
                     linewidth=0.5,
                     elinewidth=0.5)
         
      # #########
      # Finish up ... 
      plt.title('Low-pT electron performance (BParking)')
      plt.xlabel(xlabel)
      plt.ylabel('Mistag rate (w.r.t. GSF tracks, pT > 0.5 GeV)')
      plt.gca().set_yscale('log')
      ax.set_xlim(binning[0],binning[-2])
      ax.set_ylim([1.e-4, 1.])
      plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
      plt.tight_layout()
      plt.savefig(dir+'/mistag_vs_{:s}.pdf'.format(attr))
      plt.clf()
      plt.close()

   #############
   # HISTOGRAM #
   #############

   tuple = ([
      'gsf_pt',
      'gsf_mode_pt',
      'gsf_dxy',
      'gsf_dz',
      'rho',
      ],
   [
      bin_edges,
      bin_edges,
      np.linspace(0.,3.3,12),
      np.linspace(0.,22.,12),
      np.linspace(0.,44.,12),
      ],
   [
      'Transverse momentum (GeV)',
      'Mode transverse momentum (GeV)',
      'Transverse impact parameter w.r.t. beamspot (cm)',
      'Longitudinal impact parameter w.r.t. beamspot (cm)',
      'Median energy density from UE/pileup (GeV / unit area)',
      ])

   print("Histograms ...")
   for attr,binning,xlabel in zip(*tuple) :
      print(attr)

      plt.figure()
      ax = plt.subplot(111)

      has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
      has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
      has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
      has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
      curves = [
         {"label":"Low-pT (signal)","var":test[attr],"mask":(test.is_e)&(has_gsf),"condition":(has_ele),"colour":"red","fill":False,"size":8,},
         {"label":"Low-pT (bkgd)","var":test[attr],"mask":(~test.is_e)&(has_gsf),"condition":(has_ele),"colour":"red","fill":True,"size":8,},
         {"label":"EGamma (signal)","var":egamma[attr],"mask":(egamma.is_e)&(has_gsf_),"condition":(has_ele_),"colour":"purple","fill":False,"size":8,},
         {"label":"EGamma (bkgd)","var":egamma[attr],"mask":(~egamma.is_e)&(has_gsf_),"condition":(has_ele_),"colour":"purple","fill":True,"size":8,},
         ]
             
      for idx,curve in enumerate(curves) :
         # print("label:",curve["label"])
         his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=binning)
         his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=binning)
         x=binning[:-1]
         y=[ y/w if y > 0 else 0. for w,x,y in zip(bin_widths,his_passed,his_total) ]
         yhigh=[0.]*len(y)
         ylow =[0.]*len(y)
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
                     markerfacecolor = curve["colour"] if curve["fill"] else "white",
                     markersize=curve["size"],
                     linewidth=0.5,
                     elinewidth=0.5)
         
      # #########
      # Finish up ... 
      #plt.title('Low-pT electron performance (BParking)')
      plt.xlabel(xlabel)
      plt.ylabel('Denominator (w.r.t. GSF tracks, pT > 0.5 GeV)')
      ax.set_xlim(binning[0],binning[-2])
      plt.ylim([0.,None])
      plt.legend(loc='best',facecolor='white',framealpha=None,frameon=False)
      plt.tight_layout()
      plt.savefig(dir+'/denom_vs_{:s}.pdf'.format(attr))
      plt.clf()
      plt.close()
