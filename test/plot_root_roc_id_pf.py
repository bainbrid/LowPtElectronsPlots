import numpy as np
import ROOT as r
from setTDRStyle import setTDRStyle
from sklearn.metrics import roc_curve, roc_auc_score
from plot_root_common import *

################################################################################

def draw_root_roc_id_pf(
    LP,        # lowpt
    EG,        # egamma
    eta_u,     # upper
    pt_l,      # lower
    pt_u=None, # upper
    add_auc=True,
    **kwargs,
    ):
    
    # Labels for PF/EGamma
    has_gen =  EG.is_e       & (np.abs(EG.gen_eta)<eta_u)   & (EG.gen_pt>pt_l)   
    has_trk = (EG.has_trk)   & (np.abs(EG.trk_eta)<eta_u)   & (EG.trk_pt>pt_l)   
    has_gsf = (EG.has_pfgsf) & (np.abs(EG.pfgsf_eta)<eta_u) & (EG.pfgsf_pt>pt_l) 
    has_ele = (EG.has_ele)   & (np.abs(EG.ele_eta)<eta_u)   & (EG.ele_pt>pt_l)   

    # ID ROC for PF (default)
    branch = 'ele_mva_value'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(EG.is_e[has_obj],EG[branch][has_obj])
    auc = roc_auc_score(EG.is_e[has_obj],EG[branch][has_obj]) if len(set(EG.is_e[has_obj])) > 1 else 0.
    draw_root_roc(
        tpr,fpr,auc if add_auc else None,
        label='ID, PF standard',
        color=r.kMagenta+2,
        style=2)

    # ID ROC for PF (retrained)
    branch = 'ele_mva_value_retrained'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(EG.is_e[has_obj],EG[branch][has_obj])
    auc = roc_auc_score(EG.is_e[has_obj],EG[branch][has_obj]) if len(set(EG.is_e[has_obj])) > 1 else 0.
    draw_root_roc(
        tpr,fpr,auc if add_auc else None,
        label='ID, PF retrained',
        color=r.kMagenta+2)
    
################################################################################

def plot_root_roc_id_pf(lowpt,egamma,eta_upper,pt_lower,pt_upper=None):

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
    draw_root_roc_id_pf(lowpt,egamma,eta_upper,pt_lower)
    
    # Legend
    graphs = c.GetListOfPrimitives()
    xmax = 0.9
    ymin = 0.2
    legend = r.TLegend(xmax-0.35,ymin,xmax,ymin+(len(graphs)+1)*0.045)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)

    temp = r.TGraph(); temp.SetMarkerColor(r.kWhite)
    text = f"pT > {pt_lower:.1f} GeV" if pt_upper is None else f"{pt_lower:.1f} < pT < {pt_upper:.1f} GeV"
    legend.AddEntry(temp,text,"p")

    for i,gr in enumerate(graphs):
        legend.AddEntry(gr.GetName(),gr.GetName(),'l')
    legend.Draw("same")
    
    # Labels and save
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='')
    c.SaveAs(f"output/plot_root_roc_id_pf.pdf")
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='Preliminary')
    c.SaveAs(f"output/plot_root_roc_id_pf_prelim.pdf")
