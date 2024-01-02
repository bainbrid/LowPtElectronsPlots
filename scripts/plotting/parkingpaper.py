from __future__ import print_function
from __future__ import absolute_import
import builtins
import future
from future.utils import raise_with_traceback
import past
import six
import pandas as pd

import matplotlib
#print("matplotlib.get_cachedir():",matplotlib.get_cachedir())
matplotlib.use('Agg') # choose backend before doing anything else with pyplot! ('macosx')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties
# Use the stylesheet globally
plt.style.use('tdrstyle.mplstyle')

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from standalone import binomial_hpdr

import ROOT as r 
from setTDRStyle import setTDRStyle
from CMS_lumi import *

################################################################################

def cmsLabels(
        pad,lumiText,extraText=None,
        xPos=0.16,yPos=0.83,
        xPosOffset=0.1,yPosOffset=0.,
        ):

    # Latex box
    latex = r.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(r.kBlack)
    pad.cd()
    top = pad.GetTopMargin()
    right = pad.GetRightMargin()
    
    # CMS (Preliminary) label
    cmsText = "CMS"
    cmsTextFont = 61
    cmsTextSize = 0.75
    latex.SetTextFont(cmsTextFont)
    latex.SetTextSize(cmsTextSize*top)
    latex.SetTextAlign(11)
    latex.DrawLatex(xPos, yPos, cmsText)
    #pad.Update()

    if extraText != None and extraText != "":
        #extraText = "Preliminary"
        extraTextFont = 52 
        extraTextSize = 0.75*cmsTextSize
        latex.SetTextFont(extraTextFont)
        latex.SetTextSize(extraTextSize*top)
        latex.SetTextAlign(11)
        latex.DrawLatex(xPos+xPosOffset, yPos+yPosOffset, extraText)

    # CMS lumi label
    #lumiText = "33.9 fb^{-1} (13 TeV)"
    #lumiText = "scale[0.85]{"+lumiText+"}"
    lumiTextSize = 0.6
    lumiTextOffset = 0.2
    latex.SetTextFont(42)
    latex.SetTextAlign(31)
    latex.SetTextSize(lumiTextSize*top)
    latex.DrawLatex(1-right,1-top+lumiTextOffset*top,lumiText)
    #pad.Update()

################################################################################
# 
def parkingpaper(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### PARKING PAPER ##########################################################')

   option = ["default","gt2p0","gt0p5","0p5to2p0"][1]
   same_fr_wp_thr = {"default":None, "gt2p0":-2.780, "gt0p5":1.122, "0p5to2p0":None}.get(option,None)
   same_eff_wp_thr = {"default":None, "gt2p0": 5.076, "gt0p5":5.508, "0p5to2p0":None}.get(option,None)
   tight_wp_thr = {"default":None, "gt2p0": -0.427, "gt0p5":None, "0p5to2p0":None}.get(option,None)
   vtight_wp_thr = {"default":None, "gt2p0": 6.605, "gt0p5":None, "0p5to2p0":None}.get(option,None)
   vvtight_wp_thr = {"default":None, "gt2p0": 9.619, "gt0p5":None, "0p5to2p0":None}.get(option,None)
   pf_tight_wp_thr = {"default":None, "gt2p0": 0.450, "gt0p5":None, "0p5to2p0":None}.get(option,None)

   ##################################################
   # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
   # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
   
   root_file_only = True
   roc_in_root_style = True

   version = None
   pt_lower = None
   pt_upper = None

   eta_cut = 2.5
   pf_binning = 0
   idx = 1

   # Are we using only MC, or data and MC?
   only_mc = np.all(test.is_mc)

   # Reweight MC to data using kine vars
   reweight = True if only_mc else False # Reweight MC to data only if using MC
   tag = ['2023Nov14','2023Dec15','2023Dec28',][2]

   determine_weights = False # Determine new weights for MC using data
   features = {
       '2023Dec15':['log_trkpt', 'trk_eta'],
       '2023Dec28':['log_trkpt', 'rho'],
       }.get(tag)
       
   nth = 1000 # Inspect every Nth entry !!!
   
   # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   ##################################################

   # Check if weights should be applied
   if determine_weights & only_mc:
       print("CANNOT DETERMINE WEIGHTS IF USING ONLY MC!")
       quit()
   
   if pf_binning==False: # "Low-pT binning" (>0.5, >2.0, 0.5-2.0)
       version = ["gt0p5","gt2p0","0p5to2p0"][idx]
       pt_lower = {"gt0p5":0.5, "gt2p0":None,"0p5to2p0":0.5}.get(version,None)
       pt_upper = {"gt0p5":None,"gt2p0":2.0, "0p5to2p0":2.0}.get(version,None)
   else: # "PF binning" (>2.0, >5.0, 2.0-5.0, >10.0)
       version = ["gt2p0","gt5p0","2p0to5p0","gt10p0"][idx]
       pt_lower = {"gt2p0":2.0, "gt5p0":None,"2p0to5p0":2.0,"gt10p0":10.0}.get(version,None)
       pt_upper = {"gt2p0":None,"gt5p0":5.0, "2p0to5p0":5.0,"gt10p0":15.0}.get(version,None)

   print("########################################")
   print("CONFIGURATION:")
   print("  only_mc:    ",only_mc)
   print("  tag:        ",tag)
   print("  features:   ",features)
   print("  new weights:",determine_weights)
   print("  is_data:    ",~only_mc)
   print("  reweight:   ",reweight)
   print("  version:    ",version)
   print("  pf_binning: ",pf_binning)
   print("  pt_lower:   ",pt_lower)
   print("  pt_upper:   ",pt_upper)
   print("  eta_cut:    ",eta_cut)
   print("########################################")
   
   #############
   # ROC CURVE #
   #############

   plt.figure(figsize=(6,6))
   ax = plt.subplot(111)
   plt.title('Low-pT electron performance (BParking)')
   plt.xlim(1.e-4,1.)
   plt.ylim([0., 1.])
   pt_threshold = None
   if (version=="gt0p5")&~pf_binning | (version=="gt2p0")&pf_binning:
       pt_threshold = f"pT > {pt_lower:.1f} GeV"
   elif (version=="gt2p0")&~pf_binning | (version=="gt5p0")&pf_binning | (version=="gt10p0")&pf_binning:
       pt_threshold = f"pT > {pt_upper:.1f} GeV"
   elif (version=="0p5to2p0")&~pf_binning | (version=="2p0to5p0")&pf_binning:
       pt_threshold = f"{pt_lower:.1f} < pT < {pt_upper:.1f} GeV"
   else:
       print("uknown category!")
   plt.xlabel(f'Mistag rate (w.r.t. KF tracks, {pt_threshold}, |eta| < {eta_cut})')
   plt.ylabel(f'Efficiency (w.r.t. KF tracks, {pt_threshold}, |eta| < {eta_cut})')
   ax.tick_params(axis='x', pad=10.)
   plt.gca().set_xscale('log')
   plt.grid(True)
   
   ########################################
   # PLOT FOR PARKING PAPER
   ########################################

   # "by chance" line

   plt.plot(np.arange(0.,1.,plt.xlim()[0]),np.arange(0.,1.,plt.xlim()[0]),ls='dotted',lw=0.5,label="By chance")

   # MASKING
   mask = np.abs(test.trk_eta) < eta_cut
   mask2 = np.abs(egamma.trk_eta) < eta_cut
   if (version=="gt0p5")&~pf_binning | (version=="gt2p0")&pf_binning:
       pt_cut = pt_lower
       mask &= (test.trk_pt > pt_lower)
       mask2 &= (egamma.trk_pt > pt_lower)
   elif (version=="gt2p0")&~pf_binning | (version=="gt5p0")&pf_binning | (version=="gt10p0")&pf_binning:
       pt_cut = pt_upper
       mask &= (test.trk_pt > pt_upper)
       mask2 &= (egamma.trk_pt > pt_upper)
   elif (version=="0p5to2p0")&~pf_binning | (version=="2p0to5p0")&pf_binning:
       pt_cut = pt_lower
       mask &= (test.trk_pt > pt_lower) & (test.trk_pt < pt_upper)
       mask2 &= (egamma.trk_pt > pt_lower) & (egamma.trk_pt < pt_upper)
   else:
       print("uknown category!")
   #mask &= (test.gsf_pt > 0.) 
   
   test = test[mask]
   egamma = egamma[mask2]

   # Truth table (need to ignore XOR)
   # is_e,is_mc,xor,label
   #    0     0   0    0 (data for bkgd, keep)
   #    1     0   1    ? (never happens, but drop)
   #    0     1   1    ? (should ignore, but drop)
   #    1     1   0    1 (MC for signal, keep)

   # if using both data and MC, filter based on mask to keep "MC for signal" and "data for bkgd" ONLY
   if only_mc == False:
       test = test[(test.is_e == test.is_mc)]
       egamma = egamma[(egamma.is_e == egamma.is_mc)]

   # Determine MC weights vs log(trk_pt) and trk_eta
   from plotting.kmeans_reweight import kmeans_reweight, calc_weights
   if reweight==True or determine_weights==True:
       test,_ = calc_weights(test,tag=tag,reweight_features=features)
       egamma,_ = calc_weights(egamma,tag=tag,reweight_features=features)
       #kmeans_reweight(test,from_file=False) # determine weights and save to file

   # Low-pT electrons
   has_gen =  test.is_e     & (test.gen_pt>pt_cut) & (np.abs(test.gen_eta)<2.5)
   has_trk = (test.has_trk) & (test.trk_pt>pt_cut) & (np.abs(test.trk_eta)<2.5)
   has_gsf = (test.has_gsf) & (test.gsf_pt>pt_cut) & (np.abs(test.gsf_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>pt_cut) & (np.abs(test.ele_eta)<2.5)

   print("TABLE LOW PT (denom=TRK)")
   print(pd.crosstab(
       test.is_e,
       [has_trk],
       rownames=['is_e'],
       colnames=['has_trk'],
       margins=True))

   print("TABLE LOW PT (denom=ELE)")
   print(pd.crosstab(
       test.is_e,
       [has_ele],
       rownames=['is_e'],
       colnames=['has_ele'],
       margins=True))

   denom = has_gen; numer = has_trk&denom;
   trk_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test.is_e); numer = has_trk&denom;
   trk_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

   denom = has_gen&has_trk; numer = has_gsf&denom;
   gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test.is_e); numer = has_gsf&denom;
   gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

   denom = has_gen&has_trk; numer = has_ele&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test.is_e); numer = has_ele&denom; #@@
   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

   plt.plot([gsf_fr], [gsf_eff],
            marker='o', markerfacecolor='blue', markeredgecolor='blue',
            markersize=8,linestyle='none',
            label='Low-pT GSF track',
            )

   plt.plot([ele_fr], [ele_eff],
            marker='o', markerfacecolor='red', markeredgecolor='red', 
            markersize=8,linestyle='none',
            label='Low-pT electron',
            )
   
   biased_branch = 'gsf_bdtout2'
   has_obj = has_ele
   biased_fpr,biased_tpr,biased_thr = roc_curve(test.is_e[has_obj],test[biased_branch][has_obj],)
   biased_auc = roc_auc_score(test.is_e[has_obj],test[biased_branch][has_obj]) if len(set(test.is_e[has_obj])) > 1 else 0.
   plt.plot(biased_fpr*ele_fr,
            biased_tpr*ele_eff,
            linestyle='solid', color='blue', linewidth=1.0,
            label='Biased seed (AUC={:.3f})'.format(biased_auc))
   
   unbias_branch = 'gsf_bdtout1'
   has_obj = has_ele
   unbias_fpr,unbias_tpr,unbias_thr = roc_curve(test.is_e[has_obj],test[unbias_branch][has_obj])
   unbias_auc = roc_auc_score(test.is_e[has_obj],test[unbias_branch][has_obj]) if len(set(test.is_e[has_obj])) > 1 else 0.
   plt.plot(unbias_fpr*ele_fr,
            unbias_tpr*ele_eff,
            linestyle='solid', color='green', linewidth=1.0,
            label='Unbiased seed ({:.3f})'.format(unbias_auc))

   # 2020Sept15
   id_2020Sept15_branch = 'ele_mva_value_depth10'
   has_obj = has_ele
   id_2020Sept15_fpr,id_2020Sept15_tpr,id_2020Sept15_thr = roc_curve(test.is_e[has_obj],test[id_2020Sept15_branch][has_obj])
   id_2020Sept15_auc = roc_auc_score(test.is_e[has_obj],test[id_2020Sept15_branch][has_obj]) if len(set(test.is_e[has_obj])) > 1 else 0.
   plt.plot(id_2020Sept15_fpr*ele_fr,
            id_2020Sept15_tpr*ele_eff,
            linestyle='solid', color='red', linewidth=1.0,
            label='2020Sept15 ({:.3f})'.format(id_2020Sept15_auc))
   
   # 2019Aug07
   id_2019Aug07_branch = 'ele_mva_value'
   has_obj = has_ele
   id_2019Aug07_fpr,id_2019Aug07_tpr,id_2019Aug07_thr = roc_curve(test.is_e[has_obj],test[id_2019Aug07_branch][has_obj])
   id_2019Aug07_auc = roc_auc_score(test.is_e[has_obj],test[id_2019Aug07_branch][has_obj]) if len(set(test.is_e[has_obj])) > 1 else 0.
   plt.plot(id_2019Aug07_fpr*ele_fr,
            id_2019Aug07_tpr*ele_eff,
            linestyle='dashdot', color='red', linewidth=1.0,
            label='2019Aug07 ({:.3f})'.format(id_2019Aug07_auc))

   # 2021May17
   id_2021May17_branch = 'ele_mva_value_depth13'
   has_obj = has_ele
   id_2021May17_fpr,id_2021May17_tpr,id_2021May17_thr = roc_curve(test.is_e[has_obj],test[id_2021May17_branch][has_obj])
   id_2021May17_auc = roc_auc_score(test.is_e[has_obj],test[id_2021May17_branch][has_obj]) if len(set(test.is_e[has_obj])) > 1 else 0.
   plt.plot(id_2021May17_fpr*ele_fr,
            id_2021May17_tpr*ele_eff,
            linestyle='dashed', color='red', linewidth=1.0,
            label='2021May17 ({:.3f})'.format(id_2021May17_auc))

   # 2020Nov28
   id_2020Nov28_branch = 'ele_mva_value_depth11'
   has_obj = has_ele
   id_2020Nov28_fpr,id_2020Nov28_tpr,id_2020Nov28_thr = roc_curve(test.is_e[has_obj],test[id_2020Nov28_branch][has_obj])
   id_2020Nov28_auc = roc_auc_score(test.is_e[has_obj],test[id_2020Nov28_branch][has_obj]) if len(set(test.is_e[has_obj])) > 1 else 0.
   plt.plot(id_2020Nov28_fpr*ele_fr,
            id_2020Nov28_tpr*ele_eff,
            linestyle='dotted', color='red', linewidth=1.0,
            label='2020Nov28 ({:.3f})'.format(id_2020Nov28_auc))

   # PF electron
   has_pfgen   =  egamma.is_e       & (egamma.gen_pt>pt_cut)   & (np.abs(egamma.gen_eta)<2.5)
   has_pftrk   = (egamma.has_trk)   & (egamma.trk_pt>pt_cut)   & (np.abs(egamma.trk_eta)<2.5)
   has_pfgsf   = (egamma.has_pfgsf) & (egamma.pfgsf_pt>pt_cut) & (np.abs(egamma.pfgsf_eta)<2.5)
   has_pfele   = (egamma.has_ele)   & (egamma.ele_pt>pt_cut)   & (np.abs(egamma.ele_eta)<2.5)

   print("TABLE PF ALGO (denom=TRK)")
   print(pd.crosstab(
       egamma.is_e,
       [has_pftrk],
       rownames=['is_e'],
       colnames=['has_pftrk'],
       margins=True))

   print("TABLE PF ALGO (denom=ELE)")
   print(pd.crosstab(
       egamma.is_e,
       [has_pfele],
       rownames=['is_e'],
       colnames=['has_pfele'],
       margins=True))

   denom = has_pfgen; numer = has_pftrk&denom
   pftrk_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_pftrk&(~egamma.is_e); numer = has_pftrk&denom
   #pftrk_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   pftrk_fr = float(egamma.weight[numer].sum()) / float(egamma.weight[denom].sum()) \
     if float(egamma.weight[denom].sum()) > 0. else 0.

   denom = has_pfgen&has_pftrk; numer = has_pfgsf&denom
   pfgsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_pftrk&(~egamma.is_e); numer = has_pfgsf&denom
   #pfgsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   pfgsf_fr = float(egamma.weight[numer].sum()) / float(egamma.weight[denom].sum()) \
     if float(egamma.weight[denom].sum()) > 0. else 0.

   denom = has_pfgen&has_pftrk; numer = has_pfele&denom
   pfele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_pftrk&(~egamma.is_e); numer = has_pfele&denom;
   #pfele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   pfele_fr = float(egamma.weight[numer].sum()) / float(egamma.weight[denom].sum()) \
     if float(egamma.weight[denom].sum()) > 0. else 0.
   
   plt.plot([pfgsf_fr], [pfgsf_eff],
            marker='o', color='orange', 
            markersize=10, linestyle='none',
            label='EGamma seed')

   plt.plot([pfele_fr], [pfele_eff],
            marker='o', color='purple', 
            markersize=8, linestyle='none',
            label='PF electron')

   #@@ IMPORT PICKLE FILE WITH ROCS FROM SEBASTIAN SIMPLE SCRIPT !!!
   sebestian_simple_check = False
   (id_pf_retrain_fpr,id_pf_retrain_tpr,id_pf_retrain_thr,id_pf_retrain_auc,id_pf_fpr,id_pf_tpr,id_pf_thr,id_pf_auc) = (None,None,None,None,None,None,None,None)
   if sebestian_simple_check:
       import pickle
       f = open('id_pf_retrain.pkl','rb') # created by sebastian_simple.py
       data = pickle.load(f)
       (id_pf_retrain_fpr,id_pf_retrain_tpr,id_pf_retrain_thr,id_pf_retrain_auc,id_pf_fpr,id_pf_tpr,id_pf_thr,id_pf_auc) = data
       f.close()
   
#   print("TEST")
#   weighted = np.abs(egamma.weight-1.) > 1.e-6
#   sig = (egamma.is_e == egamma.is_mc)
#   print(pd.crosstab(
#       egamma.is_mc,
#       weighted,
#       rownames=['is_mc?'],
#       colnames=['weighted?'],
#       margins=True))
#   quit()
       
   # PF ID (default)
   id_pf_branch = 'ele_mva_value'
   has_pfobj = has_pfele
   if not sebestian_simple_check:
       id_pf_fpr,id_pf_tpr,id_pf_thr = roc_curve(
           egamma.is_e[has_pfobj],
           egamma[id_pf_branch][has_pfobj], 
           sample_weight=egamma.weight[has_pfobj],
          )
       id_pf_auc = roc_auc_score(
           egamma.is_e[has_pfobj],
           egamma[id_pf_branch][has_pfobj],
           sample_weight=egamma.weight[has_pfobj],
           ) if len(set(egamma.is_e[has_pfobj])) > 1 else 0.
   plt.plot(id_pf_fpr*pfele_fr,
            id_pf_tpr*pfele_eff,
            linestyle='dotted', color='purple', linewidth=1.0,
            label='PF default ID ({:.3f})'.format(id_pf_auc))

   # PF ID (retrained)
   id_pf_retrain_branch = 'ele_mva_value_retrained'
   has_pfobj = has_pfele
   if not sebestian_simple_check:
       id_pf_retrain_fpr,id_pf_retrain_tpr,id_pf_retrain_thr = roc_curve(
           egamma.is_e[has_pfobj],
           egamma[id_pf_retrain_branch][has_pfobj],
           sample_weight=egamma.weight[has_pfobj],
           )
       id_pf_retrain_auc = roc_auc_score(
           egamma.is_e[has_pfobj],
           egamma[id_pf_retrain_branch][has_pfobj],
           sample_weight=egamma.weight[has_pfobj],
           ) if len(set(egamma.is_e[has_pfobj])) > 1 else 0.
   plt.plot(id_pf_retrain_fpr*pfele_fr, 
            id_pf_retrain_tpr*pfele_eff,
            linestyle='solid', color='purple', linewidth=1.0,
            label='PF retrained ID ({:.3f})'.format(id_pf_retrain_auc))
   
   print(f"PERFORMANCE: LP eff= {ele_eff:4.2f} LP FR= {ele_fr:6.4f} PF eff= {pfele_eff:4.2f} PF FR= {pfele_fr:6.4f} PFGSF eff= {pfgsf_eff:4.2f} PFGSF FR= {pfgsf_fr:6.4f}")

   # Same-FR working point (same fake rate)

   print("Candidate WP:")
   print("PF ele, eff:",f"{pfele_eff:6.4f}")
   print("PF ele, FR :",f"{pfele_fr :6.4f}")
   print("Low-pT, eff:",f"{ele_eff  :6.4f}")
   print("Low-pT, FR: ",f"{ele_fr   :6.4f}")

   # Same-FR working point (same fake rate)
   if same_fr_wp_thr is None:
       same_fr_wp_idx = np.abs(id_2020Sept15_fpr*ele_fr-pfele_fr).argmin()
   else:
       print(f"WARNING!!! WARNING!!! WARNING!!! same_fr_wp_thr set to value = {same_fr_wp_thr}")
       same_fr_wp_idx = np.abs(id_2020Sept15_thr-same_fr_wp_thr).argmin()
   same_fr_wp_fr  = id_2020Sept15_fpr[same_fr_wp_idx]*ele_fr
   same_fr_wp_eff = id_2020Sept15_tpr[same_fr_wp_idx]*ele_eff
   same_fr_wp_thr = id_2020Sept15_thr[same_fr_wp_idx]
   same_fr_wp = test[id_2020Sept15_branch][has_ele]>same_fr_wp_thr

   print("Same-FR WP (i.e. same fake rate):")
   print("PF ele, FR :",f"{pfele_fr    :6.4f}")
   print("Low-pT, FR :",f"{same_fr_wp_fr :6.4f}")
   print("Low-pT, eff:",f"{same_fr_wp_eff:6.4f}")
   print("Low-pT, idx:",f"{same_fr_wp_idx:6.0f}")
   print("Low-pT, thr:",f"{same_fr_wp_thr:6.3f}")

   # Same-Eff working point (same efficiency)
   if same_eff_wp_thr is None:
       same_eff_wp_idx = np.abs(id_2020Sept15_tpr*ele_eff-pfele_eff).argmin()
   else:
       print(f"WARNING!!! WARNING!!! WARNING!!! same_eff_wp_thr set to value = {same_eff_wp_thr}")
       same_eff_wp_idx = np.abs(id_2020Sept15_thr-same_eff_wp_thr).argmin()
   same_eff_wp_fr  = id_2020Sept15_fpr[same_eff_wp_idx]*ele_fr
   same_eff_wp_eff = id_2020Sept15_tpr[same_eff_wp_idx]*ele_eff
   same_eff_wp_thr = id_2020Sept15_thr[same_eff_wp_idx]
   same_eff_wp = test[id_2020Sept15_branch][has_ele]>same_eff_wp_thr

   print("Same-Eff WP (i.e. same efficiency):")
   print("PF ele, eff:",f"{pfele_eff   :6.4f}")
   print("Low-pT, eff:",f"{same_eff_wp_eff:6.4f}")
   print("Low-pT, FR :",f"{same_eff_wp_fr :6.4f}")
   print("Low-pT, idx:",f"{same_eff_wp_idx:6.0f}")
   print("Low-pT, thr:",f"{same_eff_wp_thr:6.3f}")

   # Tight working point (1E-2 fake rate)
   if tight_wp_thr is None:
       pfele_fr_temp = 1.e-2
       tight_wp_idx = np.abs(id_2020Sept15_fpr*ele_fr-pfele_fr_temp).argmin()
   else:
       print(f"WARNING!!! WARNING!!! WARNING!!! tight_wp_thr set to value = {tight_wp_thr}")
       tight_wp_idx = np.abs(id_2020Sept15_thr-tight_wp_thr).argmin()
   tight_wp_fr  = id_2020Sept15_fpr[tight_wp_idx]*ele_fr
   tight_wp_eff = id_2020Sept15_tpr[tight_wp_idx]*ele_eff
   tight_wp_thr = id_2020Sept15_thr[tight_wp_idx]
   tight_wp = test[id_2020Sept15_branch][has_ele]>tight_wp_thr

   print("Tight WP (i.e. 1E-2 fake rate):")
   print("PF ele, eff:",f"{pfele_eff   :6.4f}")
   print("Low-pT, eff:",f"{tight_wp_eff:6.4f}")
   print("Low-pT, FR :",f"{tight_wp_fr :6.4f}")
   print("Low-pT, idx:",f"{tight_wp_idx:6.0f}")
   print("Low-pT, thr:",f"{tight_wp_thr:6.3f}")
   
   # Very tight working point (1E-3 fake rate)
   if vtight_wp_thr is None:
       pfele_fr_temp = 1.e-3
       vtight_wp_idx = np.abs(id_2020Sept15_fpr*ele_fr-pfele_fr_temp).argmin()
   else:
       print(f"WARNING!!! WARNING!!! WARNING!!! vtight_wp_thr set to value = {vtight_wp_thr}")
       vtight_wp_idx = np.abs(id_2020Sept15_thr-vtight_wp_thr).argmin()
   vtight_wp_fr  = id_2020Sept15_fpr[vtight_wp_idx]*ele_fr
   vtight_wp_eff = id_2020Sept15_tpr[vtight_wp_idx]*ele_eff
   vtight_wp_thr = id_2020Sept15_thr[vtight_wp_idx]
   vtight_wp = test[id_2020Sept15_branch][has_ele]>vtight_wp_thr

   print("Very tight WP (i.e. 1E-3 fake rate):")
   print("PF ele, eff:",f"{pfele_eff   :6.4f}")
   print("Low-pT, eff:",f"{vtight_wp_eff:6.4f}")
   print("Low-pT, FR :",f"{vtight_wp_fr :6.4f}")
   print("Low-pT, idx:",f"{vtight_wp_idx:6.0f}")
   print("Low-pT, thr:",f"{vtight_wp_thr:6.3f}")

   # Very very tight working point (1E-4 fake rate)
   if vvtight_wp_thr is None:
       pfele_fr_temp = 1.e-4
       vvtight_wp_idx = np.abs(id_2020Sept15_fpr*ele_fr-pfele_fr_temp).argmin()
   else:
       print(f"WARNING!!! WARNING!!! WARNING!!! vvtight_wp_thr set to value = {vvtight_wp_thr}")
       vvtight_wp_idx = np.abs(id_2020Sept15_thr-vvtight_wp_thr).argmin()
   vvtight_wp_fr  = id_2020Sept15_fpr[vvtight_wp_idx]*ele_fr
   vvtight_wp_eff = id_2020Sept15_tpr[vvtight_wp_idx]*ele_eff
   vvtight_wp_thr = id_2020Sept15_thr[vvtight_wp_idx]
   vvtight_wp = test[id_2020Sept15_branch][has_ele]>vvtight_wp_thr

   print("Very very tight WP (i.e. 1E-4 fake rate):")
   print("PF ele, eff:",f"{pfele_eff   :6.4f}")
   print("Low-pT, eff:",f"{vvtight_wp_eff:6.4f}")
   print("Low-pT, FR :",f"{vvtight_wp_fr :6.4f}")
   print("Low-pT, idx:",f"{vvtight_wp_idx:6.0f}")
   print("Low-pT, thr:",f"{vvtight_wp_thr:6.3f}")

   plt.plot(
       [same_fr_wp_fr], [same_fr_wp_eff],
       marker='*', color='red', 
       markersize=8, linestyle='none',
       label=f'Same-FR ID WP ({same_fr_wp_thr:.2f})')

   plt.plot(
       [same_eff_wp_fr], [same_eff_wp_eff],
       marker='*', color='red', 
       markersize=8, linestyle='none',
       label=f'Same-Eff ID WP ({same_eff_wp_thr:.2f})')

   plt.plot(
       [tight_wp_fr], [tight_wp_eff],
       marker='*', color='red', 
       markersize=8, linestyle='none',
       label=f'Tight ID WP ({tight_wp_thr:.2f})')

   plt.plot(
       [vtight_wp_fr], [vtight_wp_eff],
       marker='*', color='red', 
       markersize=8, linestyle='none',
       label=f'VTight ID WP ({vtight_wp_thr:.2f})')

   plt.plot(
       [vvtight_wp_fr], [vvtight_wp_eff],
       marker='*', color='red', 
       markersize=8, linestyle='none',
       label=f'VVTight ID WP ({vvtight_wp_thr:.2f})')

   # Tight working point (1E-2 fake rate) <--- For PF electrons!!! 
   if pf_tight_wp_thr is None:
       pfele_fr_temp = 1.e-2
       pf_tight_wp_idx = np.abs(id_pf_retrain_fpr*pfele_fr-pfele_fr_temp).argmin()
   else:
       print(f"WARNING!!! WARNING!!! WARNING!!! tight_wp_thr set to value = {tight_wp_thr}")
       pf_tight_wp_idx = np.abs(id_pf_retrain_thr-pf_tight_wp_thr).argmin()
   pf_tight_wp_fr  = id_pf_retrain_fpr[pf_tight_wp_idx]*pfele_fr
   pf_tight_wp_eff = id_pf_retrain_tpr[pf_tight_wp_idx]*pfele_eff
   pf_tight_wp_thr = id_pf_retrain_thr[pf_tight_wp_idx]
   pf_tight_wp = egamma[id_pf_retrain_branch][has_pfele]>pf_tight_wp_thr

   print("PF Tight WP (i.e. 1E-2 fake rate):")
   print("PF ele, eff:",f"{pfele_eff   :6.4f}")
   print("PF ele, eff:",f"{pf_tight_wp_eff:6.4f}")
   print("PF ele, FR :",f"{pf_tight_wp_fr :6.4f}")
   print("PF ele, idx:",f"{pf_tight_wp_idx:6.0f}")
   print("PF ele, thr:",f"{pf_tight_wp_thr:6.3f}")

   plt.plot(
       [pf_tight_wp_fr], [pf_tight_wp_eff],
       marker='*', color='purple', 
       markersize=8, linestyle='none',
       label=f'PF Tight ID WP ({pf_tight_wp_thr:.2f})')

   # Finish up ... 

   plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
   plt.tight_layout()
   plt.xlim(1.e-4,1.)
   plt.ylim([0., 1.])
   plt.gca().set_xscale('log')
   print('Saving pdf: '+dir+'/roc.pdf')
   plt.savefig(dir+'/roc.pdf')
   plt.clf()
   plt.close()
   
   ########################################
   # IN ROOT
   ########################################

   if not roc_in_root_style: quit()

   setTDRStyle()
   W = 800
   H = 600
   H_ref = 600
   W_ref = 800
   T = 0.08*H_ref
   B = 0.14*H_ref 
   L = 0.12*W_ref
   R = 0.05*W_ref

   c = r.TCanvas()
   c.SetLeftMargin( L/W )
   c.SetRightMargin( R/W )
   c.SetTopMargin( T/H )
   c.SetBottomMargin( B/H )
   #r.gStyle.SetPalette(r.kGreenRedViolet)
   #r.TColor.InvertPalette()
   c.SetLogx()
   
   xmin = 1.e-4
   chance_tpr = np.array(np.arange(0.,1.,xmin)[1:]) # ignore first entry @0.0
   chance_fpr = np.array(np.arange(0.,1.,xmin)[1:]) # ignore first entry @0.0
   print(chance_tpr)
   g_chance = r.TGraph(len(chance_fpr), chance_fpr, chance_tpr)
   g_chance.SetTitle("")
   g_chance.GetYaxis().SetNdivisions(505)
   g_chance.GetXaxis().SetLimits(xmin,1.)
   g_chance.GetYaxis().SetRangeUser(0.,1.)
   g_chance.SetLineStyle(2)
   g_chance.SetLineWidth(2)
   g_chance.SetLineColor(r.kGray+1)
   g_chance.Draw("AL")
   g_chance.GetXaxis().SetTitle("Misidentification probability")
   g_chance.GetYaxis().SetTitle("Efficiency")

   # LOW PT ELECTRONS

   rel_eff = ele_eff/gsf_eff
   rel_fr = ele_fr/gsf_fr

   unbias_tpr_slim = np.array(unbias_tpr[::nth]) # ignore final entry @1.0; the every Nth entry
   unbias_fpr_slim = np.array(unbias_fpr[::nth]) # ignore final entry @1.0; the every Nth entry
   g_unbias = r.TGraph(len(unbias_fpr_slim), unbias_fpr_slim*ele_fr, unbias_tpr_slim*ele_eff)
   g_unbias.SetTitle("")
   #g_unbias.SetTitle("AUC = {:.2f}".format(unbias_auc))
   g_unbias.SetLineStyle(3)
   g_unbias.SetLineWidth(2)
   g_unbias.SetLineColor(r.kGreen+3)
   #g_unbias.Draw("Lsame")

   biased_tpr_slim = np.array(biased_tpr[::nth]) # ignore final entry @1.0; the every Nth entry
   biased_fpr_slim = np.array(biased_fpr[::nth]) # ignore final entry @1.0; the every Nth entry
   g_biased = r.TGraph(len(biased_fpr_slim), biased_fpr_slim*ele_fr, biased_tpr_slim*ele_eff)
   #g_biased.SetTitle("AUC = {:.2f}".format(biased_auc))
   g_biased.SetLineStyle(5)
   g_biased.SetLineWidth(2)
   g_biased.SetLineColor(r.kGreen+3)
   #g_biased.Draw("Lsame")

   nth = 100 # "Tuned" to give smooth low-pT ID ROC ...
   tpr = id_2020Sept15_tpr
   fpr = id_2020Sept15_fpr
   id_tpr_slim = np.array(tpr[::nth]) # ignore final entry @1.0; the every Nth entry
   id_fpr_slim = np.array(fpr[::nth]) # ignore final entry @1.0; the every Nth entry
   g_id = r.TGraph(len(id_fpr_slim), id_fpr_slim*ele_fr, id_tpr_slim*ele_eff)
   #g_id.SetTitle("AUC = {:.2f}".format(id_auc))
   g_id.SetLineStyle(1)
   g_id.SetLineWidth(2)
   g_id.SetLineColor(r.kBlue)
   g_id.Draw("Lsame")

   m_ele = r.TGraph()
   m_ele.SetPoint(0,ele_fr,ele_eff)
   m_ele.SetMarkerStyle(20)
   m_ele.SetMarkerSize(2)
   m_ele.SetMarkerColor(r.kBlue)
   m_ele.Draw("Psame")

   # PF ELECTRONS

   tpr = id_pf_tpr
   fpr = id_pf_fpr
   id_tpr_slim = np.array(tpr[::nth]) # ignore final entry @1.0; the every Nth entry
   id_fpr_slim = np.array(fpr[::nth]) # ignore final entry @1.0; the every Nth entry
   g_pf1_id = r.TGraph(len(id_fpr_slim), id_fpr_slim*pfele_fr, id_tpr_slim*pfele_eff)
   #g_pf1_id = r.TGraph(len(id_fpr_slim), id_fpr_slim, id_tpr_slim)
   #g_id_pf.SetTitle("AUC = {:.2f}".format(id_auc))
   g_pf1_id.SetLineStyle(2)
   g_pf1_id.SetLineWidth(2)
   g_pf1_id.SetLineColor(r.kRed)
   #g_pf1_id.Draw("Lsame")

   import pickle
   f = open('id_pf_retrain.pkl','rb') # created by sebastian_simple.py
   data = pickle.load(f)
   (id_pf_retrain_fpr,id_pf_retrain_tpr,id_pf_retrain_thr,id_pf_retrain_auc,id_pf_fpr,id_pf_tpr,id_pf_thr,id_pf_auc) = data
   f.close()
    
   tpr = id_pf_retrain_tpr
   fpr = id_pf_retrain_fpr
   id_tpr_slim = np.array(tpr[::10]) # ignore final entry @1.0; the every Nth entry
   id_fpr_slim = np.array(fpr[::10]) # ignore final entry @1.0; the every Nth entry
   g_id_pf = r.TGraph(len(id_fpr_slim), id_fpr_slim*pfele_fr, id_tpr_slim*pfele_eff)
   #g_id_pf = r.TGraph(len(id_fpr_slim), id_fpr_slim, id_tpr_slim)
   #g_id_pf.SetTitle("AUC = {:.2f}".format(id_auc))
   g_id_pf.SetLineStyle(1)
   g_id_pf.SetLineWidth(2)
   g_id_pf.SetLineColor(r.kRed)
   g_id_pf.Draw("Lsame")

   m_pfele = r.TGraph()
   m_pfele.SetPoint(0,pfele_fr,pfele_eff)
   m_pfele.SetMarkerStyle(21)
   m_pfele.SetMarkerSize(2)
   m_pfele.SetMarkerColor(r.kRed)
   m_pfele.Draw("Psame")

   print(f"PERFORMANCE: LP eff= {ele_eff:4.2f} LP FR= {ele_fr:6.4f} PF eff= {pfele_eff:4.2f} PF FR= {pfele_fr:6.4f}")

   ##################
   # Working points #
   ##################

   # Same-FR working point (same fake rate)
   if same_fr_wp_thr is None:
       same_fr_wp_idx = np.abs(id_2020Sept15_fpr*ele_fr-pfele_fr).argmin()
   else:
       print(f"WARNING!!! WARNING!!! WARNING!!! same_fr_wp_thr set to value = {same_fr_wp_thr}")
       same_fr_wp_idx = np.abs(id_2020Sept15_thr-same_fr_wp_thr).argmin()
   same_fr_wp_fr  = id_2020Sept15_fpr[same_fr_wp_idx]*ele_fr
   same_fr_wp_eff = id_2020Sept15_tpr[same_fr_wp_idx]*ele_eff
   same_fr_wp_thr = id_2020Sept15_thr[same_fr_wp_idx]
   same_fr_wp = test[id_2020Sept15_branch][has_ele]>same_fr_wp_thr

   print("Same-FR WP (i.e. same fake rate):")
   print("PF ele, FR :",f"{pfele_fr    :6.4f}")
   print("Low-pT, FR :",f"{same_fr_wp_fr :6.4f}")
   print("Low-pT, eff:",f"{same_fr_wp_eff:6.4f}")
   print("Low-pT, idx:",f"{same_fr_wp_idx:6.0f}")
   print("Low-pT, thr:",f"{same_fr_wp_thr:6.3f}")

   m_same_fr_wp = r.TGraph()
   m_same_fr_wp.SetPoint(0,same_fr_wp_fr,same_fr_wp_eff)
   m_same_fr_wp.SetMarkerStyle(24)
   m_same_fr_wp.SetMarkerSize(2)
   m_same_fr_wp.SetMarkerColor(r.kBlue)
   #m_same_fr_wp.Draw("Psame") # <-------- Same-FR WP: Low-pT FR == PF FR

   # Same-Eff working point (same efficiency)
   if same_eff_wp_thr is None:
       same_eff_wp_idx = np.abs(id_2020Sept15_tpr*ele_eff-pfele_eff).argmin()
   else:
       print(f"WARNING!!! WARNING!!! WARNING!!! same_eff_wp_thr set to value = {same_eff_wp_thr}")
       same_eff_wp_idx = np.abs(id_2020Sept15_thr-same_eff_wp_thr).argmin()
   same_eff_wp_fr  = id_2020Sept15_fpr[same_eff_wp_idx]*ele_fr
   same_eff_wp_eff = id_2020Sept15_tpr[same_eff_wp_idx]*ele_eff
   same_eff_wp_thr = id_2020Sept15_thr[same_eff_wp_idx]
   same_eff_wp = test[id_2020Sept15_branch][has_ele]>same_eff_wp_thr

   print("Same-Eff WP (i.e. same efficiency):")
   print("PF ele, eff:",f"{pfele_eff   :6.4f}")
   print("Low-pT, eff:",f"{same_eff_wp_eff:6.4f}")
   print("Low-pT, FR :",f"{same_eff_wp_fr: 6.4f}")
   print("Low-pT, idx:",f"{same_eff_wp_idx:6.0f}")
   print("Low-pT, thr:",f"{same_eff_wp_thr:6.3f}")

   same_fr_wp_idx = np.abs(id_2020Sept15_fpr*ele_fr-pfele_fr).argmin()
   same_fr_wp_fr  = id_2020Sept15_fpr[same_fr_wp_idx]*ele_fr
   same_fr_wp_eff = id_2020Sept15_tpr[same_fr_wp_idx]*ele_eff
   same_fr_wp_thr = id_2020Sept15_thr[same_fr_wp_idx]
   same_fr_wp = test[id_2020Sept15_branch][has_ele]>same_fr_wp_thr

   m_same_eff_wp = r.TGraph()
   m_same_eff_wp.SetPoint(0,same_eff_wp_fr,same_eff_wp_eff)
   m_same_eff_wp.SetMarkerStyle(24)
   m_same_eff_wp.SetMarkerSize(2)
   m_same_eff_wp.SetMarkerColor(r.kRed)
   #m_same_eff_wp.Draw("Psame") # <-------- Same-Eff WP: Low-pT eff == PF eff

   ##########
   # Legend #
   ##########

   temp = r.TGraph()
   temp.SetMarkerColor(r.kWhite)
   text = ["p_{T} > 0.5 GeV","p_{T} > 2 GeV","0.5 < p_{T} < 2 GeV"][idx] if not pf_binning else ["p_{T} > 2 GeV","p_{T} > 5 GeV","2 < p_{T} < 5 GeV","p_{T} > 10 GeV"][idx]
   legend = r.TLegend(0.45,0.2,0.8,0.2+6*0.055)
   legend.SetTextFont(42)
   legend.SetTextSize(0.04)
   legend.AddEntry(temp,text+f", |#eta| < {eta_cut}","p")
   legend.AddEntry(m_ele,"Low-p_{T} electron cand.","p")
   legend.AddEntry(g_id,"Low-p_{T} identification","l")
   #legend.AddEntry(g_biased,"Low-p_{T} seed (biased)","l")
   #legend.AddEntry(g_unbias,"Low-p_{T} seed (unbiased)","l")
   #legend.AddEntry(m_pfele,"PF electron","p")
   legend.AddEntry(m_pfele,"PF electron candidate","p")
   #legend.AddEntry(g_pf1_id,"PF ID default","l")
   legend.AddEntry(g_id_pf,"PF identification","l")
   #legend.AddEntry(g_id,"Identification BDT","l")
   legend.AddEntry(g_chance,"By chance","l")
   legend.Draw("same")
   g_chance.Draw("same")
   
   lumiText='2018 (13 TeV)'
   extraText='Simulation'
   cmsLabels(c,lumiText,extraText=extraText)
   c.SaveAs(f"{dir}/roc_root_{version}.pdf")

   #c.Update()
   #c.RedrawAxis()
   #c.GetFrame().Draw()

   lumiText='2018 (13 TeV)'
   extraText='Simulation Preliminary'
   cmsLabels(c,lumiText,extraText=extraText)#,xPosOffset=0.,yPosOffset=-0.05)
   c.SaveAs(f"{dir}/roc_root_{version}_prelim.pdf")

   ########################################
   # END (PLOT FOR PARKING PAPER)
   ########################################

   if True:
       path = "../output/plots_train2/parkingpaper/"
       for signal,algo,title,label,binning,data in [
            (True,"PF", "gen_pt","PF GEN pT [GeV]",(100,0.,10.),egamma.gen_pt[has_pfgen]),
            (True,"PF","gen_eta","PF GEN eta",(100,-5.,5.),egamma.gen_eta[has_pfgen]),
            (True,"PF", "trk_pt","PF TRK pT [GeV]",(100,0.,10.),egamma.trk_pt[has_pfgen]),
            (False,"PF", "trk_pt","PF TRK pT [GeV]",(100,0.,10.),egamma.trk_pt[has_pftrk&~egamma.is_e]),
            (True,"PF","trk_eta","PF TRK eta",(100,-5.,5.),egamma.trk_eta[has_pfgen]),
            (False,"PF","trk_eta","PF TRK eta",(100,-5.,5.),egamma.trk_eta[has_pftrk&~egamma.is_e]),
            (True,"PF", "id_old","PF ID score (old)",(50,-10.,10.),egamma.ele_mva_value[has_pfgen]),
            (False,"PF", "id_old","PF ID score (old)",(50,-10.,10.),egamma.ele_mva_value[has_pfele&~egamma.is_e]),
            (True,"PF", "id_new","PF ID score (new)",(50,-10.,10.),egamma.ele_mva_value_retrained[has_pfgen]),
            (False,"PF", "id_new","PF ID score (new)",(50,-10.,10.),egamma.ele_mva_value_retrained[has_pfele&~egamma.is_e]),
            (True,"LP", "gen_pt","LP GEN pT [GeV]",(100,0.,10.),test.gen_pt[has_gen]),
            (True,"LP","gen_eta","LP GEN eta",(100,-5.,5.),test.gen_eta[has_gen]),
            (True,"LP", "trk_pt","LP TRK pT [GeV]",(100,0.,10.),test.trk_pt[has_gen]),
            (False,"LP", "trk_pt","LP TRK pT [GeV]",(100,0.,10.),test.trk_pt[has_trk&~test.is_e]),
            (True,"LP","trk_eta","LP TRK eta",(100,-5.,5.),test.trk_eta[has_gen]),
            (False,"LP","trk_eta","LP TRK eta",(100,-5.,5.),test.trk_eta[has_trk&~test.is_e]),
            (True,"LP", "id_old","LP BDT score (2019Aug07)",(100,-10.,10.),test.ele_mva_value[has_gen]),
            (False,"LP", "id_old","LP BDT score (2019Aug07)",(100,-10.,10.),test.ele_mva_value[has_trk&~test.is_e]),
            (True,"LP", "id_new","LP BDT score (2020Sept15)",(100,-10.,10.),test.ele_mva_value_depth10[has_gen]),
            (False,"LP", "id_new","LP BDT score (2020Sept15)",(100,-10.,10.),test.ele_mva_value_depth10[has_trk&~test.is_e]),
            #
            (True,"PF", "gen_pt_denom","PF GEN pT [GeV]",      (20,0.,10.),egamma.gen_pt[has_pfgen&has_pftrk]),
            (True,"PF", "gen_pt_numer","PF GEN pT [GeV]",      (20,0.,10.),egamma.gen_pt[has_pfgen&has_pftrk&has_pfele]),
            (True,"PF", "gen_pt_numer_tight","PF GEN pT [GeV]",(20,0.,10.),egamma.gen_pt[has_pfgen&has_pftrk&has_pfele&pf_tight_wp]),
            #
            (True,"LP", "gen_pt_denom","LP GEN pT [GeV]",      (20,0.,10.),test.gen_pt[has_gen&has_trk]),
            (True,"LP", "gen_pt_numer","LP GEN pT [GeV]",      (20,0.,10.),test.gen_pt[has_gen&has_trk&has_ele]),
            (True,"LP", "gen_pt_numer_same_fr","LP GEN pT [GeV]",(20,0.,10.),test.gen_pt[has_gen&has_trk&has_ele&same_fr_wp]),
            (True,"LP", "gen_pt_numer_same_eff","LP GEN pT [GeV]",(20,0.,10.),test.gen_pt[has_gen&has_trk&has_ele&same_eff_wp]),
            (True,"LP", "gen_pt_numer_tight","LP GEN pT [GeV]",(20,0.,10.),test.gen_pt[has_gen&has_trk&has_ele&tight_wp]),
            (True,"LP", "gen_pt_numer_vtight","LP GEN pT [GeV]",(20,0.,10.),test.gen_pt[has_gen&has_trk&has_ele&vtight_wp]),
            (True,"LP", "gen_pt_numer_vvtight","LP GEN pT [GeV]",(20,0.,10.),test.gen_pt[has_gen&has_trk&has_ele&vvtight_wp]),
            ]:
           c = r.TCanvas()
           suffix = "S" if signal else "B"
           his = r.TH1F(algo+"_"+title+"_"+suffix,"",*binning)
           for val in data: his.Fill(val if val < binning[2] else binning[2]-1.e-6)
           his.GetXaxis().SetTitle(title)
           his.SetLineWidth(2)
           his.SetLineColor(r.kGreen+3 if signal else r.kRed)
           his.Draw("")
           filename = path+title+"_"+algo+"_"+suffix+".pdf"
           root_file = ( "numer" in filename or "denom" in filename )
           if root_file_only == True:
               if root_file:
                   c.SaveAs(filename)
                   his.SaveAs(filename.replace(".pdf",".root"))
           else:
               c.SaveAs(filename)
               if root_file:
                   his.SaveAs(filename.replace(".pdf",".root"))

   
   print()
   print("################################################################################")
   print(f"PT THRESHOLDS ARE: {pt_lower} < pT < {pt_upper} GeV (pt_cut = {pt_cut})")
   print("################################################################################")

   ################################################################################
   # EFF AND MISTAG CURVES
   ################################################################################
   
   # Binning 
   bin_edges = np.linspace(0., 10., 21, endpoint=True)
   #bin_edges = np.linspace(0., 4., 8, endpoint=False)
   #bin_edges = np.append( bin_edges, np.linspace(4., 8., 4, endpoint=False) )
   #bin_edges = np.append( bin_edges, np.linspace(8., 12., 3, endpoint=True) )
   bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
   bin_widths = (bin_edges[1:] - bin_edges[:-1])
   bin_width = bin_widths[0]
   bin_widths /= bin_width

   tuple = ([
      'gen_pt',
      'trk_pt',
#      'gsf_mode_pt',
#      'gsf_dxy',
#      'gsf_dz',
#      'rho',
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

   # EFF CURVES

   print("Efficiency curves ...")
   for attr,binning,xlabel in zip(*tuple) :
      print(attr)

      plt.figure()
      ax = plt.subplot(111)

      curves = [
         #{"label":"Low-pT GSF track","var":test[attr],"mask":(test.is_e)&(has_trk),"condition":(has_gsf),"colour":"purple","fill":True,"size":8,},
         #{"label":"LP electron","var":test[attr],"mask":(test.is_e)&(has_trk),"condition":(has_ele),"colour":"blue","fill":False,"size":8,},
         {"label":"LP ele, Seed WP","var":test[attr],"mask":(test.is_e)&(has_gen)&(has_trk),"condition":(has_ele),"colour":"blue","fill":True,"size":8,},
         {"label":"LP ele, Same-FR WP","var":test[attr],"mask":(test.is_e)&(has_gen)&(has_trk),"condition":(has_ele)&(same_fr_wp),"colour":"blue","fill":False,"size":8,},
         {"label":"LP ele, Same-Eff WP","var":test[attr],"mask":(test.is_e)&(has_gen)&(has_trk),"condition":(has_ele)&(same_eff_wp),"colour":"red","fill":False,"size":8,},
         {"label":"PF electron","var":egamma[attr],"mask":(egamma.is_e)&(has_pfgen)&(has_pftrk),"condition":(has_pfele),"colour":"red","fill":True,"size":8,},
         ]
             
      for idx,curve in enumerate(curves) :
          his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=binning)
          his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=binning)
          x=binning[:-1]
          y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ]
          yhigh=[ binomial_hpdr(p,t)[1]-(p/t) if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
          ylow =[ (p/t)-binomial_hpdr(p,t)[0] if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
          yerr =[ylow,yhigh]
          #label=curve["label"]
          label='{:s} (mean={:5.3f})'.format(
              curve["label"],
              float(his_passed.sum())/float(his_total.sum()) \
              if his_total.sum() > 0 else 0.)
          ax.errorbar(
              x=x,
              y=y,
              yerr=yerr,
              #color=None,
              label=label,
              marker=curve.get("marker",'o'),
              color=curve["colour"],
              markerfacecolor = curve["colour"] if curve["fill"] else "white",
              markersize=curve["size"],
              markeredgewidth=1.0,
              linewidth=0.5,
              elinewidth=0.5
              )
          
      # #########
      # Finish up ... 
      plt.title('Low-pT electron performance (BParking)')
      plt.xlabel(xlabel)
      plt.ylabel('Efficiency (w.r.t. KF tracks, pT > 0.5 GeV)')
      ax.set_xlim(binning[0],binning[-2])
      plt.ylim([0., 1.])
      plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
      plt.tight_layout()
      plt.savefig(dir+'/eff_vs_{:s}.pdf'.format(attr))
      plt.clf()
      plt.close()

   # MISTAG CURVES #

   print("Mistag curves ...")
   for attr,binning,xlabel in zip(*tuple) :
      print(attr)

      plt.figure()
      ax = plt.subplot(111)

      curves = [
#         {"label":"Low-pT GSF track","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_gsf),"colour":"red","fill":True,"size":8,},
#         {"label":"Low-pT electron","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_ele),"colour":"blue","fill":True,"size":8,},

#{"label":"Low-pT electron","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_ele)&(same_fr_wp),"colour":"blue","fill":True,"size":8,},
#{"label":"PF electron","var":egamma[attr],"mask":(~egamma.is_e)&(has_pftrk),"condition":(has_pfele),"colour":"red","fill":True,"size":8,},

         {"label":"LP ele, Seed WP","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_ele),"colour":"blue","fill":True,"size":8,},
         {"label":"LP ele, Same-FR WP","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_ele)&(same_fr_wp),"colour":"blue","fill":False,"size":8,},
         {"label":"LP ele, Same-Eff WP","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_ele)&(same_eff_wp),"colour":"red","fill":False,"size":8,},
         {"label":"PF electron","var":egamma[attr],"mask":(~egamma.is_e)&(has_pftrk),"condition":(has_pfele),"colour":"red","fill":True,"size":8,},

          ]
   
      for idx,curve in enumerate(curves) :
         his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=binning)
         his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=binning)
         x=binning[:-1]
         y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ]
         yhigh=[ binomial_hpdr(p,t)[1]-(p/t) if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
         ylow =[ (p/t)-binomial_hpdr(p,t)[0] if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
         yerr =[ylow,yhigh]
         label='{:s} (mean={:6.4f})'.format(
             curve["label"],
             float(his_passed.sum())/float(his_total.sum()) \
             if his_total.sum() > 0 else 0.)
         ax.errorbar(x=x,
                     y=y,
                     yerr=yerr,
                     #color=None,
                     label=label,
                     marker=curve.get("marker",'o'),
                     color=curve["colour"],
                     markerfacecolor = curve["colour"] if curve["fill"] else "white",
                     markersize=curve["size"],
                     markeredgewidth=1.0,
                     linewidth=0.5,
                     elinewidth=0.5)
         
      # #########
      # Finish up ... 
      plt.title('Low-pT electron performance (BParking)')
      plt.xlabel(xlabel)
      plt.ylabel('Mistag rate (w.r.t. KF tracks, pT > 0.5 GeV)')
      plt.gca().set_yscale('log')
      ax.set_xlim(binning[0],binning[-2])
      ax.set_ylim([1.e-4, 1.])
      plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
      plt.tight_layout()
      plt.savefig(dir+'/mistag_vs_{:s}.pdf'.format(attr))
      plt.clf()
      plt.close()
