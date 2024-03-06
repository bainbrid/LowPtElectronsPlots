import numpy as np
import ROOT as r
from setTDRStyle import setTDRStyle
from sklearn.metrics import roc_curve, roc_auc_score
from plot_root_roc_common import *

################################################################################

def draw_root_roc_seeds_comparison(lowpt,egamma,eta_upper,pt_lower,pt_upper=None):
    
    # Labels for low-pT
    has_gen =  lowpt.is_e     & (lowpt.gen_pt>pt_lower) & (np.abs(lowpt.gen_eta)<eta_upper)
    has_trk = (lowpt.has_trk) & (lowpt.trk_pt>pt_lower) & (np.abs(lowpt.trk_eta)<eta_upper)
    has_gsf = (lowpt.has_gsf) & (lowpt.gsf_pt>pt_lower) & (np.abs(lowpt.gsf_eta)<eta_upper)
    has_ele = (lowpt.has_ele) & (lowpt.ele_pt>pt_lower) & (np.abs(lowpt.ele_eta)<eta_upper)
    
    # Eff and fake rate for tracks (not really needed)
    #denom = has_gen; numer = has_trk&denom;
    #trk_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    #denom = has_trk&(~lowpt.is_e); numer = has_trk&denom;
    #trk_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    #draw_root_wp(trk_eff,trk_fr,label='Low-pT track',mfc='black')
    
    # Eff and fake rate for low-pT GSF tracks
    denom = has_gen&has_trk; numer = has_gsf&denom;
    gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    denom = has_trk&(~lowpt.is_e); numer = has_gsf&denom;
    gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    draw_root_wp(gsf_eff,gsf_fr,label='Low-pT electron seed',style=20,size=2,color=r.kBlue)

    # ROC for low-pT biased seed
    branch = 'gsf_bdtout2'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(lowpt.is_e[has_obj],lowpt[branch][has_obj])
    auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
    draw_root_roc(tpr,fpr,auc,gsf_eff,gsf_fr,label='Kinematic-aware seed',color=r.kBlue,style=2)
    
    # ROC for low-pT unbiased seed
    branch = 'gsf_bdtout1'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(lowpt.is_e[has_obj],lowpt[branch][has_obj])
    auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
    draw_root_roc(tpr,fpr,auc,gsf_eff,gsf_fr,label='Kinematic-agnostic seed',color=r.kBlue)
    
    # Labels for PF/EGamma
    has_gen =  egamma.is_e       & (egamma.gen_pt>pt_lower)   & (np.abs(egamma.gen_eta)<eta_upper)
    has_trk = (egamma.has_trk)   & (egamma.trk_pt>pt_lower)   & (np.abs(egamma.trk_eta)<eta_upper)
    has_gsf = (egamma.has_pfgsf) & (egamma.pfgsf_pt>pt_lower) & (np.abs(egamma.pfgsf_eta)<eta_upper)
    has_ele = (egamma.has_ele)   & (egamma.ele_pt>pt_lower)   & (np.abs(egamma.ele_eta)<eta_upper)
    
    # Eff and fake rate for tracks (not really needed)
    #denom = has_gen; numer = has_trk&denom;
    #trk_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    #denom = has_trk&(~egamma.is_e); numer = has_trk&denom;
    #trk_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    #draw_wp_mpl(trk_eff,trk_fr,label='PF track',mfc='black')
    
    # Eff and fake rate for EGamma seeds
    denom = has_gen&has_trk; numer = has_gsf&denom;
    gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    denom = has_trk&(~egamma.is_e); numer = has_gsf&denom;
    gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    draw_root_wp(gsf_eff,gsf_fr,label='EGamma seed',style=20,size=2,color=r.kOrange+1)

################################################################################

def plot_root_roc_seeds_comparison(lowpt,egamma,eta_upper,pt_lower,pt_upper=None):

    # Cosmetics
    setTDRStyle()
    W = 800
    H = 600
    H_ref = 600
    W_ref = 800
    T = 0.08*H_ref
    B = 0.14*H_ref 
    L = 0.12*W_ref
    R = 0.05*W_ref

    # Canvas
    c = r.TCanvas()
    c.SetLeftMargin( L/W )
    c.SetRightMargin( R/W )
    c.SetTopMargin( T/H )
    c.SetBottomMargin( B/H )

    # By chance graph
    xmin = 1.e-4
    chance_tpr = np.array(np.arange(0.,1.,xmin)[1:]) # ignore first entry @ 0.
    chance_fpr = np.array(np.arange(0.,1.,xmin)[1:]) # ignore first entry @ 0.
    gr_chance = r.TGraph(len(chance_fpr), chance_fpr, chance_tpr)
    gr_chance.SetLineStyle(3)
    gr_chance.SetLineWidth(2)
    gr_chance.SetLineColor(r.kGray+1)
    gr_chance.Draw("AL")
    ngraphs = 1
    
    # Axes 
    gr_chance.SetName("By chance (AUC=0.5)")
    gr_chance.SetTitle("")
    c.SetLogx()
    gr_chance.GetXaxis().SetLimits(xmin,1.)
    gr_chance.GetXaxis().SetTitle("Misidentification probability")
    gr_chance.GetYaxis().SetRangeUser(0.,1.)
    gr_chance.GetYaxis().SetNdivisions(505)
    gr_chance.GetYaxis().SetTitle("Efficiency")

    # Draw working points and ROCs (via TGraphs)
    draw_root_seeds_comparison(lowpt,egamma,eta_upper,pt_lower)
    
    # Legend
    graphs = c.GetListOfPrimitives()
    xmax = 0.9
    ymin = 0.2
    legend = r.TLegend(xmax-0.45,ymin,xmax,ymin+(len(graphs)+1)*0.045)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)
    temp = r.TGraph(); temp.SetMarkerColor(r.kWhite)
    #text = "B^{+} #rightarrow K^{+}e^{+}e^{#minus}"
    #legend.AddEntry(temp,text,"p")
    text = f"pT > {pt_lower:.1f} GeV" if pt_upper is None else f"{pt_lower:.1f} < pT < {pt_upper:.1f} GeV"
    legend.AddEntry(temp,text,"p")
    for i,gr in enumerate(graphs):
        legend.AddEntry(gr.GetName(),gr.GetName(),'l' if gr.GetN() > 1 else 'p')
    legend.Draw("same")
    
    # Labels and save
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='')
    c.SaveAs(f"plots/roc_root_seeds_comparison.pdf")
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='Preliminary')
    c.SaveAs(f"plots/roc_root_seeds_comparison_prelim.pdf")
