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
def bparking_trigger(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### BPARKING TRIGGER ##########################################################')

   print('Trigger-based subsets of test...')
   test5 = test
   test7 = test[(test.tag_pt>7.)&(np.abs(test.tag_eta)<1.5)]
   test9 = test[(test.tag_pt>9.)&(np.abs(test.tag_eta)<1.5)]
   egamma5 = egamma
   egamma7 = egamma[(egamma.tag_pt>7.)&(np.abs(egamma.tag_eta)<1.5)]
   egamma9 = egamma[(egamma.tag_pt>9.)&(np.abs(egamma.tag_eta)<1.5)]
   print('...done')

   #############
   # ROC CURVE #
   #############

   plt.figure(figsize=(6,6))
   ax = plt.subplot(111)
   plt.title('Low-pT electron performance (BParking)')
   plt.xlim(1.e-4,1.)
   plt.ylim([0., 1.])
   plt.xlabel('Mistag rate (w.r.t. KF tracks, pT > 0.5 GeV)')
   plt.ylabel('Efficiency (w.r.t. KF tracks, pT > 0.5 GeV)')
   ax.tick_params(axis='x', pad=10.)
   plt.gca().set_xscale('log')
   plt.grid(True)

   ########################################
   # "by chance" line

   plt.plot(np.arange(0.,1.,plt.xlim()[0]),np.arange(0.,1.,plt.xlim()[0]),ls='dotted',lw=0.5,label="By chance")

   ########################################
   # Low-pT electrons (incl)

   # test5
   has_trk = (test5.has_trk) & (test5.trk_pt>0.5) & (np.abs(test5.trk_eta)<2.5)
   has_ele = (test5.has_ele) & (test5.ele_pt>0.5) & (np.abs(test5.ele_eta)<2.5)
   denom = has_trk&test5.is_e; numer = has_ele&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test5.is_e); numer = has_ele&denom;
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   test5_incl_marker, = plt.plot([ele_fr], [ele_eff],
                                 marker='o', markerfacecolor='blue', markeredgecolor='blue', 
                                 markersize=8,linestyle='none',
                                 label='Low-pT electron (tag muon pT > 5 GeV, |eta| < 2.5)',
                                 )
   
   id_branch = 'ele_mva_value_depth15'
   id_fpr,id_tpr,id_score = roc_curve(test5.is_e[has_ele],test5[id_branch][has_ele])
   id_auc = roc_auc_score(test5.is_e[has_ele],test5[id_branch][has_ele]) if len(set(test5.is_e[has_ele])) > 1 else 0.
   test5_incl_roc, = plt.plot(id_fpr*ele_fr, 
                              id_tpr*ele_eff,
                              linestyle='solid', color='blue', linewidth=1.0,
                              label='ID, 2020Feb24 (AUC={:.3f})'.format(id_auc))
   
   # test7
   has_trk = (test7.has_trk) & (test7.trk_pt>0.5) & (np.abs(test7.trk_eta)<2.5)
   has_ele = (test7.has_ele) & (test7.ele_pt>0.5) & (np.abs(test7.ele_eta)<2.5)
   denom = has_trk&test7.is_e; numer = has_ele&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test7.is_e); numer = has_ele&denom;
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   test7_incl_marker, = plt.plot([ele_fr], [ele_eff],
                                marker='o', markerfacecolor='blue', markeredgecolor='blue', 
                                markersize=6,linestyle='none',
                                label='Low-pT electron (tag muon pT > 7 GeV, |eta| < 1.5)',
                                )
   
   id_branch = 'ele_mva_value_depth15'
   id_fpr,id_tpr,id_score = roc_curve(test7.is_e[has_ele],test7[id_branch][has_ele])
   id_auc = roc_auc_score(test7.is_e[has_ele],test7[id_branch][has_ele]) if len(set(test7.is_e[has_ele])) > 1 else 0.
   test7_incl_roc, = plt.plot(id_fpr*ele_fr, 
                             id_tpr*ele_eff,
                             linestyle='solid', color='blue', linewidth=1.0,
                             label='ID, 2020Feb24 (AUC={:.3f})'.format(id_auc))

   # test9
   has_trk = (test9.has_trk) & (test9.trk_pt>0.5) & (np.abs(test9.trk_eta)<2.5)
   has_ele = (test9.has_ele) & (test9.ele_pt>0.5) & (np.abs(test9.ele_eta)<2.5)
   denom = has_trk&test9.is_e; numer = has_ele&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test9.is_e); numer = has_ele&denom;
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   test9_incl_marker, = plt.plot([ele_fr], [ele_eff],
                                marker='o', markerfacecolor='blue', markeredgecolor='blue', 
                                markersize=4,linestyle='none',
                                label='Low-pT electron (tag muon pT > 9 GeV, |eta| < 1.5)',
                                )
   
   id_branch = 'ele_mva_value_depth15'
   id_fpr,id_tpr,id_score = roc_curve(test9.is_e[has_ele],test9[id_branch][has_ele])
   id_auc = roc_auc_score(test9.is_e[has_ele],test9[id_branch][has_ele]) if len(set(test9.is_e[has_ele])) > 1 else 0.
   test9_incl_roc, = plt.plot(id_fpr*ele_fr, 
                             id_tpr*ele_eff,
                             linestyle='solid', color='blue', linewidth=1.0,
                             label='ID, 2020Feb24 (AUC={:.3f})'.format(id_auc))
   
   ########################################
   # Low-pT electrons (pT > 2.0 GeV)

   # test5
   has_trk = (test5.has_trk) & (test5.trk_pt>0.5) & (np.abs(test5.trk_eta)<2.5)
   has_ele = (test5.has_ele) & (test5.ele_pt>0.5) & (np.abs(test5.ele_eta)<2.5) & (test5.ele_pt>2.0)
   denom = has_trk&test5.is_e; numer = has_ele&denom;
   ele_high_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test5.is_e); numer = has_ele&denom;
   ele_high_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   test5_high_marker, = plt.plot([ele_high_fr], [ele_high_eff],
                                marker='^', markerfacecolor='blue', markeredgecolor='blue', 
                                markersize=8,linestyle='none',
                                label='pT > 2.0 GeV',
                                )

   id_high_branch = 'ele_mva_value_depth15'
   id_high_fpr,id_high_tpr,id_high_score = roc_curve(test5.is_e[has_ele],test5[id_high_branch][has_ele])
   id_high_auc = roc_auc_score(test5.is_e[has_ele],test5[id_high_branch][has_ele]) if len(set(test5.is_e[has_ele])) > 1 else 0.
   test5_high_roc, = plt.plot(id_high_fpr*ele_high_fr, 
                             id_high_tpr*ele_high_eff,
                             linestyle='dotted', color='blue', linewidth=1.0,
                             #label='ID, 2020Feb24 (AUC={:.3f})'.format(id_high_auc)
                             )
   
   # test7
   has_trk = (test7.has_trk) & (test7.trk_pt>0.5) & (np.abs(test7.trk_eta)<2.5)
   has_ele = (test7.has_ele) & (test7.ele_pt>0.5) & (np.abs(test7.ele_eta)<2.5) & (test7.ele_pt>2.0)
   denom = has_trk&test7.is_e; numer = has_ele&denom;
   ele_high_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test7.is_e); numer = has_ele&denom;
   ele_high_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   test7_high_marker, = plt.plot([ele_high_fr], [ele_high_eff],
                                marker='^', markerfacecolor='blue', markeredgecolor='blue', 
                                markersize=6,linestyle='none',
                                label='pT > 2.0 GeV',
                                )
   
   id_high_branch = 'ele_mva_value_depth15'
   id_high_fpr,id_high_tpr,id_high_score = roc_curve(test7.is_e[has_ele],test7[id_high_branch][has_ele])
   id_high_auc = roc_auc_score(test7.is_e[has_ele],test7[id_high_branch][has_ele]) if len(set(test7.is_e[has_ele])) > 1 else 0.
   test7_high_roc, = plt.plot(id_high_fpr*ele_high_fr, 
                             id_high_tpr*ele_high_eff,
                             linestyle='dotted', color='blue', linewidth=1.0,
                             #label='ID, 2020Feb24 (AUC={:.3f})'.format(id_high_auc)
                             )
   
   # test9
   has_trk = (test9.has_trk) & (test9.trk_pt>0.5) & (np.abs(test9.trk_eta)<2.5)
   has_ele = (test9.has_ele) & (test9.ele_pt>0.5) & (np.abs(test9.ele_eta)<2.5) & (test9.ele_pt>2.0)
   denom = has_trk&test9.is_e; numer = has_ele&denom;
   ele_high_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test9.is_e); numer = has_ele&denom;
   ele_high_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   test9_high_marker, = plt.plot([ele_high_fr], [ele_high_eff],
                                marker='^', markerfacecolor='blue', markeredgecolor='blue', 
                                markersize=4,linestyle='none',
                                label='pT > 2.0 GeV',
                                )

   id_high_branch = 'ele_mva_value_depth15'
   id_high_fpr,id_high_tpr,id_high_score = roc_curve(test9.is_e[has_ele],test9[id_high_branch][has_ele])
   id_high_auc = roc_auc_score(test9.is_e[has_ele],test9[id_high_branch][has_ele]) if len(set(test9.is_e[has_ele])) > 1 else 0.
   test9_high_roc, = plt.plot(id_high_fpr*ele_high_fr, 
                             id_high_tpr*ele_high_eff,
                             linestyle='dotted', color='blue', linewidth=1.0,
                             #label='ID, 2020Feb24 (AUC={:.3f})'.format(id_high_auc)
                             )
   
   ########################################
   # Low-pT electrons (pT < 2.0 GeV)

   # test5
   has_trk = (test5.has_trk) & (test5.trk_pt>0.5) & (np.abs(test5.trk_eta)<2.5)
   has_ele = (test5.has_ele) & (test5.ele_pt>0.5) & (np.abs(test5.ele_eta)<2.5) & (test5.ele_pt<2.0)
   denom = has_trk&test5.is_e; numer = has_ele&denom;
   ele_low_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test5.is_e); numer = has_ele&denom;
   ele_low_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   test5_low_marker, = plt.plot([ele_low_fr], [ele_low_eff],
                               marker='v', markerfacecolor='blue', markeredgecolor='blue', 
                               markersize=8,linestyle='none',
                               label='pT < 2.0 GeV',
                               )
   
   id_low_branch = 'ele_mva_value_depth15'
   id_low_fpr,id_low_tpr,id_low_score = roc_curve(test5.is_e[has_ele],test5[id_low_branch][has_ele])
   id_low_auc = roc_auc_score(test5.is_e[has_ele],test5[id_low_branch][has_ele]) if len(set(test5.is_e[has_ele])) > 1 else 0.
   test5_low_roc, = plt.plot(id_low_fpr*ele_low_fr, 
                            id_low_tpr*ele_low_eff,
                            linestyle='dotted', color='blue', linewidth=1.0
                            #label='ID, 2020Feb24 (AUC={:.3f})'.format(id_low_auc)
                            )

   # test7
   has_trk = (test7.has_trk) & (test7.trk_pt>0.5) & (np.abs(test7.trk_eta)<2.5)
   has_ele = (test7.has_ele) & (test7.ele_pt>0.5) & (np.abs(test7.ele_eta)<2.5) & (test7.ele_pt<2.0)
   denom = has_trk&test7.is_e; numer = has_ele&denom;
   ele_low_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test7.is_e); numer = has_ele&denom;
   ele_low_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   test7_low_marker, = plt.plot([ele_low_fr], [ele_low_eff],
                               marker='v', markerfacecolor='blue', markeredgecolor='blue', 
                               markersize=6,linestyle='none',
                               label='pT < 2.0 GeV',
                               )
   
   id_low_branch = 'ele_mva_value_depth15'
   id_low_fpr,id_low_tpr,id_low_score = roc_curve(test7.is_e[has_ele],test7[id_low_branch][has_ele])
   id_low_auc = roc_auc_score(test7.is_e[has_ele],test7[id_low_branch][has_ele]) if len(set(test7.is_e[has_ele])) > 1 else 0.
   test7_low_roc, = plt.plot(id_low_fpr*ele_low_fr, 
                            id_low_tpr*ele_low_eff,
                            linestyle='dotted', color='blue', linewidth=1.0
                            #label='ID, 2020Feb24 (AUC={:.3f})'.format(id_low_auc)
                            )

   # test9
   has_trk = (test9.has_trk) & (test9.trk_pt>0.5) & (np.abs(test9.trk_eta)<2.5)
   has_ele = (test9.has_ele) & (test9.ele_pt>0.5) & (np.abs(test9.ele_eta)<2.5) & (test9.ele_pt<2.0)
   denom = has_trk&test9.is_e; numer = has_ele&denom;
   ele_low_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test9.is_e); numer = has_ele&denom;
   ele_low_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   test9_low_marker, = plt.plot([ele_low_fr], [ele_low_eff],
                               marker='v', markerfacecolor='blue', markeredgecolor='blue', 
                               markersize=4,linestyle='none',
                               label='pT < 2.0 GeV',
                               )
   
   id_low_branch = 'ele_mva_value_depth15'
   id_low_fpr,id_low_tpr,id_low_score = roc_curve(test9.is_e[has_ele],test9[id_low_branch][has_ele])
   id_low_auc = roc_auc_score(test9.is_e[has_ele],test9[id_low_branch][has_ele]) if len(set(test9.is_e[has_ele])) > 1 else 0.
   test9_low_roc, = plt.plot(id_low_fpr*ele_low_fr, 
                            id_low_tpr*ele_low_eff,
                            linestyle='dotted', color='blue', linewidth=1.0
                            #label='ID, 2020Feb24 (AUC={:.3f})'.format(id_low_auc)
                            )
   
   ########################################
   # EGamma PF GSF electrons

   # egamma5
   has_trk = (egamma5.has_trk) & (egamma5.trk_pt>0.5) & (np.abs(egamma5.trk_eta)<2.5)
   has_ele = (egamma5.has_ele) & (egamma5.ele_pt>0.5) & (np.abs(egamma5.ele_eta)<2.5)
   denom = has_trk&egamma5.is_e; numer = has_ele&denom
   pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~egamma5.is_e); numer = has_ele&denom
   pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   egamma5_incl_marker, = plt.plot([pf_fr], [pf_eff],
                                  marker='o', color='purple', 
                                  markersize=8, linestyle='none',
                                  label='PF electron')

   pf_id_fpr,pf_id_tpr,pf_id_score = roc_curve(egamma5.is_e[has_ele],egamma5['ele_mva_value_retrained'][has_ele])
   pf_id_auc = roc_auc_score(egamma5.is_e[has_ele],egamma5['ele_mva_value_retrained'][has_ele]) if len(set(egamma5.is_e[has_ele])) > 1 else 0.
   egamma5_incl_roc, = plt.plot(pf_id_fpr*pf_fr, 
                               pf_id_tpr*pf_eff,
                               linestyle='solid', color='purple', linewidth=1.0,
                               label='ID, retrain (AUC={:.3f})'.format(pf_id_auc))

   # egamma7
   has_trk = (egamma7.has_trk) & (egamma7.trk_pt>0.5) & (np.abs(egamma7.trk_eta)<2.5)
   has_ele = (egamma7.has_ele) & (egamma7.ele_pt>0.5) & (np.abs(egamma7.ele_eta)<2.5)
   denom = has_trk&egamma7.is_e; numer = has_ele&denom
   pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~egamma7.is_e); numer = has_ele&denom
   pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   egamma7_incl_marker, = plt.plot([pf_fr], [pf_eff],
                                  marker='o', color='purple', 
                                  markersize=6, linestyle='none',
                                  label='PF electron')

   pf_id_fpr,pf_id_tpr,pf_id_score = roc_curve(egamma7.is_e[has_ele],egamma7['ele_mva_value_retrained'][has_ele])
   pf_id_auc = roc_auc_score(egamma7.is_e[has_ele],egamma7['ele_mva_value_retrained'][has_ele]) if len(set(egamma7.is_e[has_ele])) > 1 else 0.
   egamma7_incl_roc, = plt.plot(pf_id_fpr*pf_fr, 
                               pf_id_tpr*pf_eff,
                               linestyle='solid', color='purple', linewidth=1.0,
                               label='ID, retrain (AUC={:.3f})'.format(pf_id_auc))

   # egamma9
   has_trk = (egamma9.has_trk) & (egamma9.trk_pt>0.5) & (np.abs(egamma9.trk_eta)<2.5)
   has_ele = (egamma9.has_ele) & (egamma9.ele_pt>0.5) & (np.abs(egamma9.ele_eta)<2.5)
   denom = has_trk&egamma9.is_e; numer = has_ele&denom
   pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~egamma9.is_e); numer = has_ele&denom
   pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   egamma9_incl_marker, = plt.plot([pf_fr], [pf_eff],
                                  marker='o', color='purple', 
                                  markersize=4, linestyle='none',
                                  label='PF electron')

   pf_id_fpr,pf_id_tpr,pf_id_score = roc_curve(egamma9.is_e[has_ele],egamma9['ele_mva_value_retrained'][has_ele])
   pf_id_auc = roc_auc_score(egamma9.is_e[has_ele],egamma9['ele_mva_value_retrained'][has_ele]) if len(set(egamma9.is_e[has_ele])) > 1 else 0.
   egamma9_incl_roc, = plt.plot(pf_id_fpr*pf_fr, 
                               pf_id_tpr*pf_eff,
                               linestyle='solid', color='purple', linewidth=1.0,
                               label='ID, retrain (AUC={:.3f})'.format(pf_id_auc))
   


#   ########################################
#   # Working points
#
#   id_ELE = np.abs(id_fpr*ele_fr-pf_fr).argmin()
#   same_fr = test[id_branch]>id_score[id_ELE]
#   x,y = id_fpr[id_ELE]*ele_fr,id_tpr[id_ELE]*ele_eff
#   #plt.plot([x], [y], marker='o', markerfacecolor='white', markeredgecolor='blue', markersize=8)
#   #plt.text(x, y+0.03, "WP", fontsize=8, ha='center', va='center', color='blue' )
#
#   id_high_ELE = np.abs(id_high_fpr*ele_high_fr-pf_fr).argmin()
#   same_fr_high = test[id_high_branch]>id_high_score[id_high_ELE]
#   x,y = id_high_fpr[id_high_ELE]*ele_high_fr,id_high_tpr[id_high_ELE]*ele_high_eff
#   #plt.plot([x], [y], marker='^', markerfacecolor='white', markeredgecolor='blue', markersize=8)
#   #plt.text(x, y+0.03, "WP", fontsize=8, ha='center', va='center', color='blue' )
#
#   id_low_ELE = np.abs(id_low_fpr*ele_low_fr-pf_fr).argmin()
#   same_fr_low = test[id_low_branch]>id_low_score[id_low_ELE]
#   x,y = id_low_fpr[id_low_ELE]*ele_low_fr,id_low_tpr[id_low_ELE]*ele_low_eff
#   #plt.plot([x], [y], marker='v', markerfacecolor='white', markeredgecolor='blue', markersize=8)
#   #plt.text(x, y+0.03, "WP", fontsize=8, ha='center', va='center', color='blue' )

   ##########
   # Finish up ... 
   plt.legend([test5_incl_marker,
               test5_low_marker,
               test5_high_marker,
               egamma5_incl_marker,
               test5_incl_marker,
               test7_incl_marker,
               test9_incl_marker,
               ],
              ['Low-pT electrons (incl.)',
               'Low-pT electrons (pT < 2 GeV)',
               'Low-pT electrons (pT > 2 GeV)',
               'PF electrons',
               'Tag muon: pT > 5 GeV, |eta| < 2.5',
               'Tag muon: pT > 7 GeV, |eta| < 1.5',
               'Tag muon: pT > 9 GeV, |eta| < 1.5',
               ],
              loc='upper left',facecolor='white',framealpha=None,frameon=False)
   plt.tight_layout()
   plt.savefig(dir+'/roc.pdf')
   plt.clf()
   plt.close()

#   ##############
#   # EFF CURVES #
#   ##############
#
#   # Binning 
#   bin_edges = np.linspace(0., 4., 8, endpoint=False)
#   bin_edges = np.append( bin_edges, np.linspace(4., 8., 4, endpoint=False) )
#   bin_edges = np.append( bin_edges, np.linspace(8., 12., 3, endpoint=True) )
#   bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
#   bin_widths = (bin_edges[1:] - bin_edges[:-1])
#   bin_width = bin_widths[0]
#   bin_widths /= bin_width
#   #print("bin_edges",bin_edges)
#   #print("bin_centres",bin_centres)
#   #print("bin_widths",bin_widths)
#   #print("bin_width",bin_width)
#
#   tuple = ([
#      'gen_pt',
#      'gsf_pt',
#      'gsf_mode_pt',
#      'gsf_dxy',
#      'gsf_dz',
#      'rho',
#      ],
#   [
#      bin_edges,
#      bin_edges,
#      bin_edges,
#      np.linspace(0.,3.3,12),
#      np.linspace(0.,22.,12),
#      np.linspace(0.,44.,12),
#      ],
#   [
#      'Generator-level transverse momentum (GeV)',
#      'Transverse momentum (GeV)',
#      'Mode transverse momentum (GeV)',
#      'Transverse impact parameter w.r.t. beamspot (cm)',
#      'Longitudinal impact parameter w.r.t. beamspot (cm)',
#      'Median energy density from UE/pileup (GeV / unit area)',
#      ])
#
#   print("Efficiency curves ...")
#   for attr,binning,xlabel in zip(*tuple) :
#      print(attr)
#
#      plt.figure()
#      ax = plt.subplot(111)
#
#      has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
#      has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
#      has_ele_low = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5) & (test.ele_pt<2.0)
#      has_ele_high = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5) & (test.ele_pt>2.0)
#      has_trk_ = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
#      has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
#      curves = [
#         {"label":"Low-pT electron","var":test[attr],"mask":(test.is_e)&(has_trk),"condition":(has_ele),"colour":"blue","fill":True,"size":8,},
#         {"label":"Same mistag rate","var":test[attr],"mask":(test.is_e)&(has_trk),"condition":(has_ele)&(same_fr),"colour":"blue","fill":False,"size":8,},
#         {"label":"Same mistag rate","var":test[attr],"mask":(test.is_e)&(has_trk),"condition":(has_ele_high)&(same_fr_high),"colour":"blue","fill":False,"size":8,"marker":"^"},
#         {"label":"Same mistag rate","var":test[attr],"mask":(test.is_e)&(has_trk),"condition":(has_ele_low)&(same_fr_low),"colour":"blue","fill":False,"size":8,"marker":"v"},
#         {"label":"PF electron","var":egamma[attr],"mask":(egamma.is_e)&(has_trk_),"condition":(has_ele_),"colour":"purple","fill":True,"size":8,},
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
#         label='{:s} (mean={:5.3f})'.format(curve["label"],
#                                            float(his_passed.sum())/float(his_total.sum()) \
#                                               if his_total.sum() > 0 else 0.)
#         ax.errorbar(x=x,
#                     y=y,
#                     yerr=yerr,
#                     #color=None,
#                     label=label,
#                     marker=curve.get("marker",'o'),
#                     color=curve["colour"],
#                     markerfacecolor = curve["colour"] if curve["fill"] else "white",
#                     markersize=curve["size"],
#                     linewidth=0.5,
#                     elinewidth=0.5)
#         
#      # #########
#      # Finish up ... 
#      plt.title('Low-pT electron performance (BParking)')
#      plt.xlabel(xlabel)
#      plt.ylabel('Efficiency (w.r.t. KF tracks, pT > 0.5 GeV)')
#      ax.set_xlim(binning[0],binning[-2])
#      plt.ylim([0., 1.])
#      plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
#      plt.tight_layout()
#      plt.savefig(dir+'/eff_vs_{:s}.pdf'.format(attr))
#      plt.clf()
#      plt.close()
#
#   #################
#   # MISTAG CURVES #
#   #################
#
#   print("Mistag curves ...")
#   for attr,binning,xlabel in zip(*tuple) :
#      print(attr)
#
#      plt.figure()
#      ax = plt.subplot(111)
#
#      has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
#      has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
#      has_ele_low = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5) & (test.ele_pt<2.0)
#      has_ele_high = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5) & (test.ele_pt>2.0)
#      has_trk_ = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
#      has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
#      curves = [
#         {"label":"Low-pT electron","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_ele),"colour":"blue","fill":True,"size":8,},
#         {"label":"Same mistag rate","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_ele)&(same_fr),"colour":"blue","fill":False,"size":8,},
#         {"label":"Same mistag rate","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_ele_high)&(same_fr_high),"colour":"blue","fill":False,"size":8,"marker":"^"},
#         {"label":"Same mistag rate","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_ele_low)&(same_fr_low),"colour":"blue","fill":False,"size":8,"marker":"v"},
#         {"label":"PF electron","var":egamma[attr],"mask":(~egamma.is_e)&(has_trk_),"condition":(has_ele_),"colour":"purple","fill":True,"size":8,},
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
#                     marker=curve.get("marker",'o'),
#                     color=curve["colour"],
#                     markerfacecolor = curve["colour"] if curve["fill"] else "white",
#                     markersize=curve["size"],
#                     linewidth=0.5,
#                     elinewidth=0.5)
#         
#      # #########
#      # Finish up ... 
#      plt.title('Low-pT electron performance (BParking)')
#      plt.xlabel(xlabel)
#      plt.ylabel('Mistag rate (w.r.t. KF tracks, pT > 0.5 GeV)')
#      plt.gca().set_yscale('log')
#      ax.set_xlim(binning[0],binning[-2])
#      ax.set_ylim([1.e-4, 1.])
#      plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
#      plt.tight_layout()
#      plt.savefig(dir+'/mistag_vs_{:s}.pdf'.format(attr))
#      plt.clf()
#      plt.close()
