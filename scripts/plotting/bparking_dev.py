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
def bparking_dev(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### BPARKING DEV ##########################################################')

   threshold = 1.e-1

   #############
   # ROC CURVE #
   #############

   plt.figure(figsize=(6,6))
   ax = plt.subplot(111)
   plt.title('Electrons, pT > 0.5 GeV')
   plt.xlim(1.e-4,1.)
   plt.ylim(1.e-3,1.)
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

   has_ele = (test.has_ele) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
#   denom = has_ele&test.is_e; numer = has_ele&denom;
#   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   denom = has_ele&(~test.is_e); numer = has_ele&denom;
#   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
##   plt.plot([ele_fr], [ele_eff],
##            marker='o', markerfacecolor='blue', markeredgecolor='blue', 
##            markersize=8,linestyle='none',
##            label='Low-pT electron',
##            )
#
#   id_branch = 'gsf_bdtout1'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
#   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr, 
#            id_tpr*ele_eff,
#            linestyle='solid', color='red', linewidth=1.0,
#            label='Seeding, unbiased (AUC={:.3f})'.format(id_auc))
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_seed = test[id_branch]>id_score[idx]
#
#   for threshold in [2,3,4,5,6] :
#      idx = np.abs(id_score-float(threshold)).argmin()
#      wp = test[id_branch]>id_score[idx]
#      x,y = id_fpr[idx]*ele_fr,id_tpr[idx]*ele_eff
#      plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='red', markersize=6)
#      plt.text(x, y+0.02, f"{threshold:.1f}", fontsize=6, ha='center', va='center', color='red' )
#
##   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test['training_out'][has_ele])
##   id_auc = roc_auc_score(test.is_e[has_ele],test['training_out'][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
##   plt.plot(id_fpr*ele_fr, 
##            id_tpr*ele_eff,
##            linestyle='dashed', color='blue', linewidth=1.0,
##            label='2019Aug07 on-the-fly (AUC={:.3f})'.format(id_auc))
#
#   id_branch = 'ele_mva_value'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
#   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr,
#            id_tpr*ele_eff,
#            linestyle='solid', color='blue', linewidth=1.0,
#            label='2019Aug07 (AUC={:.3f})'.format(id_auc))
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_default = test[id_branch]>id_score[idx]
#
#   for threshold in [2,3,4,5,6] :
#      idx = np.abs(id_score-float(threshold)).argmin()
#      wp = test[id_branch]>id_score[idx]
#      x,y = id_fpr[idx]*ele_fr,id_tpr[idx]*ele_eff
#      plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='blue', markersize=6)
#      plt.text(x, y+0.02, f"{threshold:.1f}", fontsize=6, ha='center', va='center', color='blue' )
#
##   idx_5 = np.abs(id_score-5.).argmin()
##   wp_5 = test[id_branch]>id_score[idx_5]
##   x,y = id_fpr[idx_5]*ele_fr,id_tpr[idx_5]*ele_eff
##   plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='black', markersize=8)
##   plt.text(x, y+0.03, "WP5", fontsize=8, ha='center', va='center', color='black' )
#
##   id_branch = 'ele_mva_value_depth10'
##   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
##   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
##   plt.plot(id_fpr*ele_fr, 
##            id_tpr*ele_eff,
##            linestyle='solid', color='green', linewidth=1.0,
##            label='depth10 (AUC={:.3f})'.format(id_auc))
##   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
##   wp_depth10 = test[id_branch]>id_score[idx]
#
##   id_branch = 'ele_mva_value_depth11'
##   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
##   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
##   plt.plot(id_fpr*ele_fr, 
##            id_tpr*ele_eff,
##            linestyle='solid', color='orange', linewidth=1.0,
##            label='depth11 (AUC={:.3f})'.format(id_auc))
##   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
##   wp_depth11 = test[id_branch]>id_score[idx]
#
#   id_branch = 'ele_mva_value_depth13'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
#   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr, 
#            id_tpr*ele_eff,
#            linestyle='solid', color='orange', linewidth=1.0,
#            label='depth=13, trees=1000 (AUC={:.3f})'.format(id_auc))
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_depth13 = test[id_branch]>id_score[idx]
#
#   for threshold in [2,3,4,5,6] :
#      idx = np.abs(id_score-float(threshold)).argmin()
#      wp = test[id_branch]>id_score[idx]
#      x,y = id_fpr[idx]*ele_fr,id_tpr[idx]*ele_eff
#      plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='orange', markersize=6)
#      plt.text(x, y+0.02, f"{threshold:.1f}", fontsize=6, ha='center', va='center', color='orange' )
#
#   id_branch = 'ele_mva_value_depth15'
#   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
#   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr, 
#            id_tpr*ele_eff,
#            linestyle='solid', color='purple', linewidth=1.0,
#            label='depth=15, trees=1000 (AUC={:.3f})'.format(id_auc))
#   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
#   wp_depth15 = test[id_branch]>id_score[idx]
#
#   for threshold in [2,3,4,5,6] :
#      idx = np.abs(id_score-float(threshold)).argmin()
#      wp = test[id_branch]>id_score[idx]
#      x,y = id_fpr[idx]*ele_fr,id_tpr[idx]*ele_eff
#      plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='purple', markersize=6)
#      plt.text(x, y+0.02, f"{threshold:.1f}", fontsize=6, ha='center', va='center', color='purple' )

   ########################################
   # Electron (pT > 2.0 GeV)

   has_high = has_ele & (test.gsf_pt>2.0)
   denom = has_ele&test.is_e; numer = has_high&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_ele&(~test.is_e); numer = has_high&denom;
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([ele_fr], [ele_eff],
            marker='^', markerfacecolor='none', markeredgecolor='blue', 
            markersize=8,linestyle='none',
            label='pT > 2.0 GeV',
            )

   id_branch = 'gsf_bdtout1'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
   id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
   plt.plot(id_fpr*ele_fr,
            id_tpr*ele_eff,
            linestyle='solid',color='red',linewidth=1.0,
            )
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_unbiased_high = test[id_branch][has_high]>id_score[idx]

   id_branch = 'ele_mva_value'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
   id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
   plt.plot(id_fpr*ele_fr,
            id_tpr*ele_eff,
            linestyle='solid',color='blue',linewidth=1.0,
            )
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_default_high = test[id_branch][has_high]>id_score[idx]

   id_branch = 'ele_mva_value_depth13'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
   id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
   plt.plot(id_fpr*ele_fr,
            id_tpr*ele_eff,
            linestyle='solid',color='orange',linewidth=1.0,
            )
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_depth13_high = test[id_branch][has_high]>id_score[idx]

   id_branch = 'ele_mva_value_depth15'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_high],test[id_branch][has_high])
   id_auc = roc_auc_score(test.is_e[has_high],test[id_branch][has_high]) if len(set(test.is_e[has_high])) > 1 else 0.
   plt.plot(id_fpr*ele_fr,
            id_tpr*ele_eff,
            linestyle='solid',color='purple',linewidth=1.0,
            )
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_depth15_high = test[id_branch][has_high]>id_score[idx]

#   wp_default = wp_default_high
#   wp_depth10 = wp_depth10_high
#   wp_depth11 = wp_depth11_high

   ########################################
   # Electron (0.5 < pT < 2.0 GeV)

   has_low = has_ele & (test.gsf_pt<2.0)
   denom = has_ele&test.is_e; numer = has_low&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_ele&(~test.is_e); numer = has_low&denom;
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   plt.plot([ele_fr], [ele_eff],
            marker='v', markerfacecolor='none', markeredgecolor='blue', 
            markersize=8,linestyle='none',
            label='0.5 < pT < 2.0 GeV',
            )

   id_branch = 'gsf_bdtout1'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
   id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
   plt.plot(id_fpr*ele_fr,
            id_tpr*ele_eff,
            linestyle='solid',color='red',linewidth=1.0,
            )
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_unbiased_low = test[id_branch][has_low]>id_score[idx]

   id_branch = 'ele_mva_value'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
   id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
   plt.plot(id_fpr*ele_fr,
            id_tpr*ele_eff,
            linestyle='solid',color='blue',linewidth=1.0,
            )
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_default_low = test[id_branch][has_low]>id_score[idx]

   id_branch = 'ele_mva_value_depth13'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
   id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
   plt.plot(id_fpr*ele_fr,
            id_tpr*ele_eff,
            linestyle='solid',color='orange',linewidth=1.0,
            )
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_depth13_low = test[id_branch][has_low]>id_score[idx]

   id_branch = 'ele_mva_value_depth15'
   id_fpr,id_tpr,id_score = roc_curve(test.is_e[has_low],test[id_branch][has_low])
   id_auc = roc_auc_score(test.is_e[has_low],test[id_branch][has_low]) if len(set(test.is_e[has_low])) > 1 else 0.
   plt.plot(id_fpr*ele_fr,
            id_tpr*ele_eff,
            linestyle='solid',color='purple',linewidth=1.0,
            )
   idx = np.abs(id_fpr*ele_fr-threshold).argmin()
   wp_depth15_low = test[id_branch][has_low]>id_score[idx]

#   wp_default = wp_default_low
#   wp_depth10 = wp_depth10_low
#   wp_depth11 = wp_depth11_low
   
   ##########
   # Finish up ... 
   plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False) # 'center right'
   plt.tight_layout()
   plt.savefig(dir+'/roc.pdf')
   plt.gca().set_xscale('log')
   plt.savefig(dir+'/roc_log.pdf')
   plt.gca().set_yscale('log')
   plt.savefig(dir+'/roc_loglog.pdf')
   plt.clf()
   plt.close()

   #########
   # SETUP #
   #########

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
#      'gsf_mode_pt',
#      'gsf_dxy',
#      'gsf_dz',
      'rho',
      ],
   [
      bin_edges,
      bin_edges,
#      bin_edges,
#      np.linspace(0.,3.3,12),
#      np.linspace(0.,22.,12),
      np.linspace(0.,44.,12),
      ],
   [
      'Generator-level transverse momentum (GeV)',
      'Transverse momentum (GeV)',
#      'Mode transverse momentum (GeV)',
#      'Transverse impact parameter w.r.t. beamspot (cm)',
#      'Longitudinal impact parameter w.r.t. beamspot (cm)',
      'Median energy density from UE/pileup (GeV / unit area)',
      ])

   ##############
   # EFF CURVES #
   ##############

#   print("Efficiency curves ...")
#   for attr,binning,xlabel in zip(*tuple) :
#      print(attr)
#
#      plt.figure()
#      ax = plt.subplot(111)
#
#      has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
#      curves = [
#         {"label":"2019Aug07","var":test[attr],"mask":(test.is_e)&(has_ele),"condition":(wp_default),"colour":"blue","size":8,},
#         {"label":"Depth10","var":test[attr],"mask":(test.is_e)&(has_ele),"condition":(wp_depth10),"colour":"orange","size":8,},
#         {"label":"Depth11","var":test[attr],"mask":(test.is_e)&(has_ele),"condition":(wp_depth11),"colour":"green","size":8,},
#         #{"label":"Seed","var":test[attr],"mask":(test.is_e)&(has_ele),"condition":(wp_seed),"colour":"red","size":8,},
#         ]
#      
#      for idx,curve in enumerate(curves) :
#         # print("label:",curve["label"])
#         his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=binning)
#         his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=binning)
#         x=binning[:-1]
#         y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ]
#         yhigh=[ binomial_hpdr(p,t)[1]-(p/t) if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
#         ylow =[ (p/t)-binomial_hpdr(p,t)[0] if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
#         yerr =[ylow,yhigh]
#         label='{:s} (mean={:5.3f})'.format(curve["label"],
#                                            float(his_passed.sum())/float(his_total.sum()) \
#                                               if his_total.sum() > 0 else 0.)
#         ax.errorbar(x=x,
#                     y=y,
#                     yerr=yerr,
#                     #color=None,
#                     label=label,
#                     marker='o',
#                     color=curve["colour"],
#                     markerfacecolor = curve["colour"] if curve.get("fill",False) else "white",
#                     markersize=curve["size"],
#                     linewidth=0.5,
#                     elinewidth=0.5)
#         
#      # #########
#      # Finish up ... 
#      plt.title('Low-pT electron performance')
#      plt.xlabel(xlabel)
#      plt.ylabel('Efficiency (w.r.t. electrons, pT > 0.5 GeV)')
#      ax.set_xlim(binning[0],binning[-2])
#      plt.ylim([0., 1.])
#      plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
#      plt.tight_layout()
#      plt.savefig(dir+'/eff_vs_{:s}.pdf'.format(attr))
#      plt.clf()
#      plt.close()

   #################
   # MISTAG CURVES #
   #################

#   print("Mistag curves ...")
#   for attr,binning,xlabel in zip(*tuple) :
#      print(attr)
#
#      plt.figure()
#      ax = plt.subplot(111)
#
#      has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
#      curves = [
#         {"label":"2019Aug07","var":test[attr],"mask":(~test.is_e)&(has_ele),"condition":(wp_default),"colour":"blue","size":8,},
#         {"label":"Depth10","var":test[attr],"mask":(~test.is_e)&(has_ele),"condition":(wp_depth10),"colour":"orange","size":8,},
#         {"label":"Depth11","var":test[attr],"mask":(~test.is_e)&(has_ele),"condition":(wp_depth11),"colour":"green","size":8,},
#         #{"label":"Seed","var":test[attr],"mask":(~test.is_e)&(has_ele),"condition":(wp_seed),"colour":"red","size":8,},
#         ]
#   
#      for idx,curve in enumerate(curves) :
#         his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=binning)
#         his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=binning)
#         x=binning[:-1]
#         y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ]
#         yhigh=[ binomial_hpdr(p,t)[1]-(p/t) if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
#         ylow =[ (p/t)-binomial_hpdr(p,t)[0] if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
#         yerr =[ylow,yhigh]
#         label='{:s} (mean={:6.4f})'.format(curve["label"],
#                                            float(his_passed.sum())/float(his_total.sum()) \
#                                               if his_total.sum() > 0 else 0.)
#         ax.errorbar(x=x,
#                     y=y,
#                     yerr=yerr,
#                     #color=None,
#                     label=label,
#                     marker='o',
#                     color=curve["colour"],
#                     markerfacecolor = curve["colour"] if curve.get("fill",False) else "white",
#                     markersize=curve["size"],
#                     linewidth=0.5,
#                     elinewidth=0.5)
#         
#      # #########
#      # Finish up ... 
#      plt.title('Low-pT electron performance')
#      plt.xlabel(xlabel)
#      plt.ylabel('Mistag rate (w.r.t. electrons, pT > 0.5 GeV)')
#      plt.gca().set_yscale('log')
#      ax.set_xlim(binning[0],binning[-2])
#      ax.set_ylim([1.e-4, 1.])
#      plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
#      plt.tight_layout()
#      plt.savefig(dir+'/mistag_vs_{:s}.pdf'.format(attr))
#      plt.clf()
#      plt.close()
