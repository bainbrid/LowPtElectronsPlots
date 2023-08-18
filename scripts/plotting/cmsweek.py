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
# 
def cmsweek(dir,test,egamma,has_pfgsf_branches=True,AxE=True) :
   print('##### CMSWEEK ##########################################################')

   version = ["gt0p5","gt2p0","0p5to2p0"][0]
   # Lower pT threshold!
   #pt_lower = {"gt0p5":0.5,"gt2p0":None,"0p5to2p0":0.5}.get(version,None)
   #pt_upper = {"gt0p5":None,"gt2p0":2.0,"0p5to2p0":2.0}.get(version,None)
   pt_lower = {"gt0p5":2.0,"gt2p0":None,"0p5to2p0":2.0}.get(version,None)
   pt_upper = {"gt0p5":None,"gt2p0":5.0,"0p5to2p0":5.0}.get(version,None)
   
   #############
   # ROC CURVE #
   #############

   plt.figure(figsize=(6,6))
   ax = plt.subplot(111)
   plt.title('Low-pT electron performance (BParking)')
   plt.xlim(1.e-4,1.)
   plt.ylim([0., 1.])
   pt_threshold = None
   if version == "gt0p5":
       pt_threshold = f"pT > {pt_lower:.1f} GeV"
   elif version == "gt2p0": 
       pt_threshold = f"pT > {pt_upper:.1f} GeV"
   elif version == "0p5to2p0": 
       pt_threshold = f"{pt_lower:.1f} < pT < {pt_upper:.1f} GeV"
   else:
       print("uknown category!")
   plt.xlabel(f'Mistag rate (w.r.t. KF tracks, {pt_threshold})')
   plt.ylabel(f'Efficiency (w.r.t. KF tracks, {pt_threshold})')
   ax.tick_params(axis='x', pad=10.)
   plt.gca().set_xscale('log')
   plt.grid(True)

   ########################################
   # PLOT FOR PARKING PAPER
   ########################################

   # "by chance" line

   plt.plot(np.arange(0.,1.,plt.xlim()[0]),np.arange(0.,1.,plt.xlim()[0]),ls='dotted',lw=0.5,label="By chance")

   # MASKING
   mask = np.abs(test.trk_eta) < 2.5
   mask2 = np.abs(egamma.trk_eta) < 2.5
   if version == "gt0p5":
       pt_cut = pt_lower
       mask &= (test.trk_pt > pt_lower)
       mask2 &= (egamma.trk_pt > pt_lower)
   elif version == "gt2p0": 
       pt_cut = pt_upper
       mask &= (test.trk_pt > pt_upper)
       mask2 &= (egamma.trk_pt > pt_upper)
   elif version == "0p5to2p0": 
       pt_cut = pt_lower
       mask &= (test.trk_pt > pt_lower) & (test.trk_pt < pt_upper)
       mask2 &= (egamma.trk_pt > pt_lower) & (egamma.trk_pt < pt_upper)
   else:
       print("uknown category!")
   #mask &= (test.gsf_pt > 0.) 

   test = test[mask]
   egamma = egamma[mask2]

   # Low-pT electrons

   has_gen =                  (test.gen_pt>pt_cut) & (np.abs(test.gen_eta)<2.5)
   has_trk = (test.has_trk) & (test.trk_pt>pt_cut) & (np.abs(test.trk_eta)<2.5)
   has_gsf = (test.has_gsf) & (test.gsf_pt>pt_cut) & (np.abs(test.gsf_eta)<2.5)
   has_ele = (test.has_ele) & (test.ele_pt>pt_cut) & (np.abs(test.ele_eta)<2.5)

   print("TABLE")
   print(pd.crosstab(
       test.is_e,
       [has_trk],
       rownames=['is_e'],
       colnames=['has_trk'],
       margins=True))

   denom = has_trk&test.is_e; numer = has_trk&denom;
   trk_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test.is_e); numer = has_trk&denom;
   trk_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

   denom = has_trk&test.is_e; numer = has_gsf&denom;
   gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test.is_e); numer = has_gsf&denom;
   gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

   denom = has_trk&test.is_e; numer = has_ele&denom;
   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_trk&(~test.is_e); numer = has_ele&denom;
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
   biased_fpr,biased_tpr,biased_thr = roc_curve(test.is_e[has_trk],test[biased_branch][has_trk])
   biased_auc = roc_auc_score(test.is_e[has_trk],test[biased_branch][has_trk]) if len(set(test.is_e[has_trk])) > 1 else 0.
   plt.plot(biased_fpr,
            biased_tpr,
            linestyle='solid', color='blue', linewidth=1.0,
            label='Seeding (AUC={:.3f})'.format(biased_auc))
   
   unbias_branch = 'gsf_bdtout1'
   unbias_fpr,unbias_tpr,unbias_thr = roc_curve(test.is_e[has_trk],test[unbias_branch][has_trk])
   unbias_auc = roc_auc_score(test.is_e[has_trk],test[unbias_branch][has_trk]) if len(set(test.is_e[has_trk])) > 1 else 0.
   plt.plot(unbias_fpr,
            unbias_tpr,
            linestyle='solid', color='green', linewidth=1.0,
            label='Unbiased ({:.3f})'.format(unbias_auc))

   # 2019Aug07
   id_branch = 'ele_mva_value'
   id_fpr,id_tpr,id_thr = roc_curve(test.is_e[has_trk],test[id_branch][has_trk])
   id_auc = roc_auc_score(test.is_e[has_trk],test[id_branch][has_trk]) if len(set(test.is_e[has_trk])) > 1 else 0.
   plt.plot(id_fpr, 
            id_tpr,
            linestyle='solid', color='red', linewidth=1.0,
            label='2019Aug07 ({:.3f})'.format(id_auc))

   # 2020Sept15
   id_2020Sept15_branch = 'ele_mva_value_depth10'
   id_2020Sept15_fpr,id_2020Sept15_tpr,id_2020Sept15_thr = roc_curve(test.is_e[has_trk],test[id_2020Sept15_branch][has_trk])
   id_2020Sept15_auc = roc_auc_score(test.is_e[has_trk],test[id_2020Sept15_branch][has_trk]) if len(set(test.is_e[has_trk])) > 1 else 0.
   plt.plot(id_2020Sept15_fpr, 
            id_2020Sept15_tpr,
            linestyle='dashed', color='red', linewidth=1.0,
            label='2020Sept15 ({:.3f})'.format(id_2020Sept15_auc))

   # 2020Nov28
   id_2020Nov28_branch = 'ele_mva_value_depth11'
   id_2020Nov28_fpr,id_2020Nov28_tpr,id_2020Nov28_thr = roc_curve(test.is_e[has_trk],test[id_2020Nov28_branch][has_trk])
   id_2020Nov28_auc = roc_auc_score(test.is_e[has_trk],test[id_2020Nov28_branch][has_trk]) if len(set(test.is_e[has_trk])) > 1 else 0.
   plt.plot(id_2020Nov28_fpr, 
            id_2020Nov28_tpr,
            linestyle='dotted', color='red', linewidth=1.0,
            label='2020Nov28 ({:.3f})'.format(id_2020Nov28_auc))

   # 2021May17
   id_2021May17_branch = 'ele_mva_value_depth13'
   id_2021May17_fpr,id_2021May17_tpr,id_2021May17_thr = roc_curve(test.is_e[has_trk],test[id_2021May17_branch][has_trk])
   id_2021May17_auc = roc_auc_score(test.is_e[has_trk],test[id_2021May17_branch][has_trk]) if len(set(test.is_e[has_trk])) > 1 else 0.
   plt.plot(id_2021May17_fpr, 
            id_2021May17_tpr,
            linestyle='dashdot', color='red', linewidth=1.0,
            label='2021May17 ({:.3f})'.format(id_2021May17_auc))

   # PF electron
   has_pfgen   =                      (egamma.gen_pt>pt_cut)   & (np.abs(egamma.gen_eta)<2.5)
   has_pftrk   = (egamma.has_trk)   & (egamma.trk_pt>pt_cut)   & (np.abs(egamma.trk_eta)<2.5)
   #has_pfgsf   = (egamma.has_gsf)   & (egamma.gsf_pt>pt_cut)   & (np.abs(egamma.gsf_eta)<2.5)
   has_pfgsf   = (egamma.has_pfgsf) & (egamma.pfgsf_pt>pt_cut) & (np.abs(egamma.pfgsf_eta)<2.5)
   has_pfele   = (egamma.has_ele)   & (egamma.ele_pt>pt_cut)   & (np.abs(egamma.ele_eta)<2.5)

   denom = has_pftrk&egamma.is_e; numer = has_pfgsf&denom
   pfgsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_pftrk&(~egamma.is_e); numer = has_pfgsf&denom
   pfgsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

   denom = has_pftrk&egamma.is_e; numer = has_pfele&denom
   pfele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   denom = has_pftrk&(~egamma.is_e); numer = has_pfele&denom
   pfele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

   plt.plot([pfgsf_fr], [pfgsf_eff],
            marker='o', color='orange', 
            markersize=10, linestyle='none',
            label='PF GSF')

   plt.plot([pfele_fr], [pfele_eff],
            marker='o', color='purple', 
            markersize=8, linestyle='none',
            label='PF electron')

   # PF ID (default)
   pf_id_branch = 'ele_mva_value'
   pf_id_fpr,pf_id_tpr,pf_id_thr = roc_curve(egamma.is_e[has_pfele],egamma[pf_id_branch][has_pfele])
   pf_id_auc = roc_auc_score(egamma.is_e[has_pfele],egamma[pf_id_branch][has_pfele]) if len(set(egamma.is_e[has_pfele])) > 1 else 0.
   plt.plot(pf_id_fpr*pfele_fr, 
            pf_id_tpr*pfele_eff,
            linestyle='dotted', color='purple', linewidth=1.0,
            label='PF default ID ({:.3f})'.format(pf_id_auc))

   # PF ID (retrained)
   pf_id_retrain_branch = 'ele_mva_value_retrained'
   pf_id_retrain_fpr,pf_id_retrain_tpr,pf_id_retrain_thr = roc_curve(egamma.is_e[has_pfele],egamma[pf_id_retrain_branch][has_pfele])
   pf_id_retrain_auc = roc_auc_score(egamma.is_e[has_pfele],egamma[pf_id_retrain_branch][has_pfele]) if len(set(egamma.is_e[has_pfele])) > 1 else 0.
   plt.plot(pf_id_retrain_fpr*pfele_fr, 
            pf_id_retrain_tpr*pfele_eff,
            linestyle='solid', color='purple', linewidth=1.0,
            label='PF retrained ID ({:.3f})'.format(pf_id_retrain_auc))
   
   print(f"PERFORMANCE: LP eff= {ele_eff:4.2f} LP FR= {ele_fr:6.4f} PF eff= {pfele_eff:4.2f} PF FR= {pfele_fr:6.4f} PFGSF eff= {pfgsf_eff:4.2f} PFGSF FR= {pfgsf_fr:6.4f}")

   # Finish up ... 

   plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
   plt.tight_layout()
   print('Saving pdf: '+dir+'/roc.pdf')
   plt.savefig(dir+'/roc.pdf')
   plt.clf()
   plt.close()

   ########################################
   # IN ROOT
   ########################################

   setTDRStyle()
   W = 800
   H = 600
   H_ref = 600
   W_ref = 800
   T = 0.08*H_ref
   B = 0.14*H_ref 
   L = 0.12*W_ref
   R = 0.04*W_ref

   c = r.TCanvas()
   c.SetLeftMargin( L/W )
   c.SetRightMargin( R/W )
   c.SetTopMargin( T/H )
   c.SetBottomMargin( B/H )
   #r.gStyle.SetPalette(r.kGreenRedViolet)
   #r.TColor.InvertPalette()
   c.SetLogx()
   
   rel_eff = ele_eff/gsf_eff
   rel_fr = ele_fr/gsf_fr

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
   g_chance.SetLineColor(r.kGray)
   g_chance.Draw("AL")
   g_chance.GetXaxis().SetTitle("Mistag probability")
   g_chance.GetYaxis().SetTitle("Signal efficiency")

   nth = 100 # Inspect every Nth entry !!! 
   unbias_tpr_slim = np.array(unbias_tpr[:-1][::nth]) # ignore final entry @1.0; the every Nth entry
   unbias_fpr_slim = np.array(unbias_fpr[:-1][::nth]) # ignore final entry @1.0; the every Nth entry
   g_unbias = r.TGraph(len(unbias_fpr_slim), unbias_fpr_slim*rel_fr, unbias_tpr_slim*rel_eff)
   g_unbias.SetTitle("")
   #g_unbias.SetTitle("AUC = {:.2f}".format(unbias_auc))
   g_unbias.SetLineStyle(5)
   g_unbias.SetLineWidth(3)
   g_unbias.SetLineColor(r.kGreen+3)
   #g_unbias.Draw("Lsame")

   biased_tpr_slim = np.array(biased_tpr[:-1][::nth]) # ignore final entry @1.0; the every Nth entry
   biased_fpr_slim = np.array(biased_fpr[:-1][::nth]) # ignore final entry @1.0; the every Nth entry
   g_biased = r.TGraph(len(biased_fpr_slim), biased_fpr_slim*rel_fr, biased_tpr_slim*rel_eff)
   #g_biased.SetTitle("AUC = {:.2f}".format(biased_auc))
   g_biased.SetLineStyle(3)
   g_biased.SetLineWidth(3)
   g_biased.SetLineColor(r.kGreen+3)
   #g_biased.Draw("Lsame")

   tpr = id_2020Sept15_tpr
   fpr = id_2020Sept15_fpr
   id_tpr_slim = np.array(tpr[:-1][::nth]) # ignore final entry @1.0; the every Nth entry
   id_fpr_slim = np.array(fpr[:-1][::nth]) # ignore final entry @1.0; the every Nth entry
   g_id = r.TGraph(len(id_fpr_slim), id_fpr_slim, id_tpr_slim)
   #g_id.SetTitle("AUC = {:.2f}".format(id_auc))
   g_id.SetLineStyle(1)
   g_id.SetLineWidth(2)
   g_id.SetLineColor(r.kBlue)
   #g_id.Draw("Lsame")

   m_ele = r.TGraph()
   m_ele.SetPoint(0,ele_fr,ele_eff)
   m_ele.SetMarkerStyle(20)
   m_ele.SetMarkerSize(2)
   m_ele.SetMarkerColor(r.kBlue)
   #m_ele.Draw("Psame")

   tpr = pf_id_tpr
   fpr = pf_id_fpr
   id_tpr_slim = np.array(tpr[:-1][::nth]) # ignore final entry @1.0; the every Nth entry
   id_fpr_slim = np.array(fpr[:-1][::nth]) # ignore final entry @1.0; the every Nth entry
   #g_pf_id = r.TGraph(len(id_fpr_slim), id_fpr_slim*pfele_fr, id_tpr_slim*pfele_eff)
   g_pf1_id = r.TGraph(len(id_fpr_slim), id_fpr_slim, id_tpr_slim)
   #g_pf_id.SetTitle("AUC = {:.2f}".format(id_auc))
   g_pf1_id.SetLineStyle(2)
   g_pf1_id.SetLineWidth(2)
   g_pf1_id.SetLineColor(r.kRed)
   g_pf1_id.Draw("Lsame")

   tpr = pf_id_retrain_tpr
   fpr = pf_id_retrain_fpr
   id_tpr_slim = np.array(tpr[:-1][::nth]) # ignore final entry @1.0; the every Nth entry
   id_fpr_slim = np.array(fpr[:-1][::nth]) # ignore final entry @1.0; the every Nth entry
   #g_pf_id = r.TGraph(len(id_fpr_slim), id_fpr_slim*pfele_fr, id_tpr_slim*pfele_eff)
   g_pf_id = r.TGraph(len(id_fpr_slim), id_fpr_slim, id_tpr_slim)
   #g_pf_id.SetTitle("AUC = {:.2f}".format(id_auc))
   g_pf_id.SetLineStyle(1)
   g_pf_id.SetLineWidth(2)
   g_pf_id.SetLineColor(r.kRed)
   g_pf_id.Draw("Lsame")

   m_pfele = r.TGraph()
   m_pfele.SetPoint(0,pfele_fr,pfele_eff)
   m_pfele.SetMarkerStyle(21)
   m_pfele.SetMarkerSize(2)
   m_pfele.SetMarkerColor(r.kRed)
   #m_pfele.Draw("Psame")

   print(f"PERFORMANCE: LP eff= {ele_eff:4.2f} LP FR= {ele_fr:6.4f} PF eff= {pfele_eff:4.2f} PF FR= {pfele_fr:6.4f}")
   
   # Working points

   unbias_wp = np.abs(unbias_fpr_slim*rel_fr-pfele_fr).argmin()
   unbias_wp_fr   = unbias_fpr_slim[unbias_wp]
   unbias_wp_eff  = unbias_tpr_slim[unbias_wp]

   id_wp = np.abs(id_fpr-pfele_fr).argmin()
   id_wp_fr   = id_fpr[id_wp]
   id_wp_eff  = id_tpr[id_wp]
   print(type(test[id_branch][has_trk]))
   print(type(id_thr[id_wp]))
   id_same_fr = test[id_branch][has_trk]>id_thr[id_wp]
   print("WP")
   print(pfele_fr)
   print(unbias_wp_fr)
   print(id_wp_fr)
   print(pfele_eff)
   print(unbias_wp_eff)
   print(id_wp_eff)

   m_unbias_wp = r.TGraph()
   m_unbias_wp.SetPoint(0,unbias_wp_fr,unbias_wp_eff)
   m_unbias_wp.SetMarkerStyle(24)
   m_unbias_wp.SetMarkerSize(2)
   m_unbias_wp.SetMarkerColor(r.kBlue)
   #m_unbias_wp.Draw("Psame")

   m_id_wp = r.TGraph()
   m_id_wp.SetPoint(0,id_wp_fr,id_wp_eff)
   m_id_wp.SetMarkerStyle(24)
   m_id_wp.SetMarkerSize(2)
   m_id_wp.SetMarkerColor(r.kBlue)
   #m_id_wp.Draw("Psame")

   legend = r.TLegend(0.55,0.2,0.9,0.2+5*0.05)
   legend.SetTextFont(42)
   legend.SetTextSize(0.04)
   legend.AddEntry(m_ele,"Low-p_{T} reconstruction","p")
   legend.AddEntry(g_id,"Low-p_{T} identification","l")
   #legend.AddEntry(m_pfele,"PF electron","p")
   legend.AddEntry(m_pfele,"PF reconstruction","p")
   legend.AddEntry(g_pf1_id,"PF ID default","l")
   legend.AddEntry(g_pf_id,"PF identification","l")
   #legend.AddEntry(g_id,"Identification BDT","l")
   #legend.AddEntry(g_biased,"Seeding BDT (biased)","l")
   #legend.AddEntry(g_unbias,"Seeding BDT (unbiased)","l")
   legend.AddEntry(g_chance,"By chance","l")
   legend.Draw("same")

   CMS_lumi( c, 4, 11 )
   c.Update()
   c.RedrawAxis()
   c.GetFrame().Draw()
   c.SaveAs(f"{dir}/roc_root_{version}.pdf")

   ########################################
   # END (PLOT FOR PARKING PAPER)
   ########################################

   path = "../output/plots_train2/cmsweek/"
   for algo,title,label,binning,data in [
           ("PF", "gen_pt","PF GEN pT [GeV]",(100,0.,10.),egamma.gen_pt),
           ("PF","gen_eta","PF GEN eta",(100,-5.,5.),egamma.gen_eta),
           ("PF", "trk_pt","PF TRK pT [GeV]",(100,0.,10.),egamma.trk_pt),
           ("PF","trk_eta","PF TRK eta",(100,-5.,5.),egamma.trk_eta),
           ("PF", "id_old","PF ID score (old)",(100,-10.,10.),egamma.ele_mva_value),
           ("PF", "id_new","PF ID score (new)",(100,-10.,10.),egamma.ele_mva_value_retrained),
           ("LP", "gen_pt","LP GEN pT [GeV]",(100,0.,10.),test.gen_pt),
           ("LP","gen_eta","LP GEN eta",(100,-5.,5.),test.gen_eta),
           ("LP", "trk_pt","LP TRK pT [GeV]",(100,0.,10.),test.trk_pt),
           ("LP","trk_eta","LP TRK eta",(100,-5.,5.),test.trk_eta),
           ("LP", "id_old","LP BDT score (2019Aug07)",(100,-10.,10.),test.ele_mva_value),
           ("LP", "id_new","LP BDT score (2020Sept15)",(100,-10.,10.),test.ele_mva_value_depth10),
           ]:
       c = r.TCanvas()
       his = r.TH1F(algo+"_"+title,"",*binning)
       for val in data: his.Fill(val)
       his.GetXaxis().SetTitle(title)
       his.Draw("")
       filename = path+title+"_"+algo+".pdf"
       c.SaveAs(filename)
   
   print("QUITTING BEFORE EFF PLOTS !!!")
   quit()
   


#   ########################################
#   # GSF track (pT > 0.5 GeV, VL WP for Seed BDT)
#
#   has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
#   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
#   denom = has_trk&test.is_e; numer = has_gsf&denom;
#   gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   denom = has_trk&(~test.is_e); numer = has_gsf&denom;
#   gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   plt.plot([gsf_fr], [gsf_eff],
#            marker='o', markerfacecolor='red', markeredgecolor='red', 
#            markersize=8,linestyle='none',
#            label='Low-pT GSF track',
#            )
#
#   unb_branch = 'gsf_bdtout1'
#   unb_fpr,unb_tpr,unb_thr = roc_curve(test.is_e[has_gsf],test[unb_branch][has_gsf])
#   unb_auc = roc_auc_score(test.is_e[has_gsf],test[unb_branch][has_gsf]) if len(set(test.is_e[has_gsf])) > 1 else 0.
#   plt.plot(unb_fpr*gsf_fr, 
#            unb_tpr*gsf_eff,
#            linestyle='solid', color='red', linewidth=1.0,
#            label='Seeding (AUC={:.3f})'.format(unb_auc))
#
#   ########################################
#   # Electron (pT > 0.5 GeV, VL WP for Seed BDT) 
#
#   has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
#   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
#   denom = has_trk&test.is_e; numer = has_ele&denom;
#   ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   denom = has_trk&(~test.is_e); numer = has_ele&denom;
#   ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   plt.plot([ele_fr], [ele_eff],
#            marker='o', markerfacecolor='blue', markeredgecolor='blue', 
#            markersize=8,linestyle='none',
#            label='Low-pT electron',
#            )
#
#   id_branch = 'ele_mva_value_depth15'
#   id_fpr,id_tpr,id_thr = roc_curve(test.is_e[has_ele],test[id_branch][has_ele])
#   id_auc = roc_auc_score(test.is_e[has_ele],test[id_branch][has_ele]) if len(set(test.is_e[has_ele])) > 1 else 0.
#   plt.plot(id_fpr*ele_fr, 
#            id_tpr*ele_eff,
#            linestyle='solid', color='blue', linewidth=1.0,
#            label='ID, 2020Feb24 (AUC={:.3f})'.format(id_auc))
#
#   ########################################
#   # EGamma PF GSF electrons
#
#   has_trk = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
#   has_ele = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
#   denom = has_trk&egamma.is_e; numer = has_ele&denom
#   pf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   denom = has_trk&(~egamma.is_e); numer = has_ele&denom
#   pf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
#   plt.plot([pf_fr], [pf_eff],
#            marker='o', color='purple', 
#            markersize=8, linestyle='none',
#            label='PF electron')
#
#   pf_id_fpr,pf_id_tpr,pf_id_thr = roc_curve(egamma.is_e[has_ele],egamma['ele_mva_value_retrained'][has_ele])
#   pf_id_auc = roc_auc_score(egamma.is_e[has_ele],egamma['ele_mva_value_retrained'][has_ele]) if len(set(egamma.is_e[has_ele])) > 1 else 0.
#   plt.plot(pf_id_fpr*pf_fr, 
#            pf_id_tpr*pf_eff,
#            linestyle='solid', color='purple', linewidth=1.0,
#            label='ID, retrained (AUC={:.3f})'.format(pf_id_auc))
#
#   ##########
#   # Finish up ... 
#   plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
#   plt.tight_layout()
#   plt.savefig(dir+'/roc.pdf')
#   plt.clf()
#   plt.close()

   ##############
   # EFF CURVES #
   ##############

   #id_ELE = np.abs(id_fpr*ele_fr-pf_fr).argmin()
   #same_fr = test[id_branch]>id_thr[id_ELE]
   
   # Binning 
   bin_edges = np.linspace(0., 4., 8, endpoint=False)
   bin_edges = np.append( bin_edges, np.linspace(4., 8., 4, endpoint=False) )
   bin_edges = np.append( bin_edges, np.linspace(8., 12., 3, endpoint=True) )
   bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
   bin_widths = (bin_edges[1:] - bin_edges[:-1])
   bin_width = bin_widths[0]
   bin_widths /= bin_width

   tuple = ([
      'gen_pt',
#      'gsf_pt',
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

   print("Efficiency curves ...")
   for attr,binning,xlabel in zip(*tuple) :
      print(attr)

      plt.figure()
      ax = plt.subplot(111)

#      has_trk = (test.has_trk) & (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.5)
#      has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
#      has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
#      has_ele_T = (has_ele) & (test.gsf_bdtout1>3.05)
#      has_trk_ = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
#      has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
#      has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
      curves = [
         #{"label":"Low-pT GSF track","var":test[attr],"mask":(test.is_e)&(has_trk),"condition":(has_gsf),"colour":"purple","fill":True,"size":8,},
         {"label":"Low-p_{T} electron","var":test[attr],"mask":(test.is_e)&(has_trk),"condition":(has_ele),"colour":"blue","fill":True,"size":8,},
         {"label":"Low-p_{T} electron","var":test[attr],"mask":(test.is_e)&(has_trk),"condition":(has_ele)&(id_same_fr),"colour":"blue","fill":False,"size":8,},
         {"label":"PF electron","var":egamma[attr],"mask":(egamma.is_e)&(has_pftrk),"condition":(has_pfele),"colour":"red","fill":True,"size":8,},
         ]
             
      for idx,curve in enumerate(curves) :
         his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=binning)
         his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=binning)
         x=binning[:-1]
         y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ]
         yhigh=[ binomial_hpdr(p,t)[1]-(p/t) if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
         ylow =[ (p/t)-binomial_hpdr(p,t)[0] if t > 0 else 0. for p,t in zip(his_passed,his_total) ]
         yerr =[ylow,yhigh]
         label=curve["label"]
#         label='{:s} (mean={:5.3f})'.format(curve["label"],
#                                            float(his_passed.sum())/float(his_total.sum()) \
#                                               if his_total.sum() > 0 else 0.)
         ax.errorbar(x=x,
                     y=y,
                     yerr=yerr,
                     #color=None,
                     label=label,
                     marker=curve.get("marker",'o'),
                     color=curve["colour"],
                     markerfacecolor = curve["colour"] if curve["fill"] else "white",
                     markersize=curve["size"],
                     linewidth=0.5,
                     elinewidth=0.5)
         
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
#      has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.5)
#      has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.5)
#      has_ele_T = (has_ele) & (test.gsf_bdtout1>3.05)
#      has_trk_ = (egamma.has_trk) & (egamma.trk_pt>0.5) & (np.abs(egamma.trk_eta)<2.5)
#      has_gsf_ = (egamma.has_gsf) & (egamma.gsf_pt>0.5) & (np.abs(egamma.gsf_eta)<2.5)
#      has_ele_ = (egamma.has_ele) & (egamma.ele_pt>0.5) & (np.abs(egamma.ele_eta)<2.5)
#      curves = [
#         {"label":"Low-pT GSF track","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_gsf),"colour":"red","fill":True,"size":8,},
#         {"label":"Low-pT electron","var":test[attr],"mask":(~test.is_e)&(has_trk),"condition":(has_ele),"colour":"blue","fill":True,"size":8,},
#         {"label":"Same mistag rate","var":test[attr],"mask":(test.is_e)&(has_trk),"condition":(has_ele)&(same_fr),"colour":"blue","fill":False,"size":8,},
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
