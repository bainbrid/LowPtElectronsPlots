import numpy as np
import ROOT as r
from setTDRStyle import setTDRStyle
from sklearn.metrics import roc_curve, roc_auc_score
from plot_root_common import *

################################################################################

def draw_root_score_id_pf(
    LP,        # lowpt
    EG,        # egamma
    eta_u,     # upper
    pt_l,      # lower
    pt_u=None, # upper
    add_auc=True,
    **kwargs,
    ): 
    # Labels for PF/EGamma
    has_gen =  EG.is_e       & (EG.gen_pt>pt_l)   & (np.abs(EG.gen_eta)<eta_u)
    has_trk = (EG.has_trk)   & (EG.trk_pt>pt_l)   & (np.abs(EG.trk_eta)<eta_u)
    has_gsf = (EG.has_pfgsf) & (EG.pfgsf_pt>pt_l) & (np.abs(EG.pfgsf_eta)<eta_u)
    has_ele = (EG.has_ele)   & (EG.ele_pt>pt_l)   & (np.abs(EG.ele_eta)<eta_u)

    nbin=60
    xmin=-20.
    xmax=10.
    ymin=0.
    ymax=0.15
    xtitle='PF electron ID, BDT score'
    xunit=None

    # ID ROC for PF (default)
    branch = 'ele_mva_value'
    has_obj = has_ele
    s = EG[branch][has_gen&has_obj]
    b = EG[branch][~EG.is_e&has_obj]
    
    draw_root_score(
        s,
        title='Signal (default)',
        nbin=nbin,xmin=xmin,xmax=xmax,
        ymin=ymin,ymax=ymax,
        xtitle=xtitle,xunit=xunit,
        style=2,width=2,color=r.kGreen+1,
        )

    draw_root_score(
        b,
        title='Bkgd (default)',
        nbin=nbin,xmin=xmin,xmax=xmax,
        ymin=ymin,ymax=ymax,
        xtitle=xtitle,xunit=xunit,
        style=2,width=2,color=r.kRed,
        same=True
        )

    # ID ROC for PF (retrained)
    branch = 'ele_mva_value_retrained'
    has_obj = has_ele
    s = EG[branch][has_gen&has_obj]
    b = EG[branch][~EG.is_e&has_obj]
    
    draw_root_score(
        s,
        title='Signal (new)',
        nbin=nbin,xmin=xmin,xmax=xmax,
        ymin=ymin,ymax=ymax,
        xtitle=xtitle,xunit=xunit,
        style=1,width=2,color=r.kGreen+1,
        same=True
        )

    draw_root_score(
        b,
        title='Bkgd (new)',
        nbin=nbin,xmin=xmin,xmax=xmax,
        xtitle=xtitle,xunit=xunit,
        style=1,width=2,color=r.kRed,
        same=True
        )

################################################################################

def plot_root_score_id_pf(lowpt,egamma,eta_upper,pt_lower,pt_upper=None):

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

    # Draw working points and ROCs (via TGraphs)
    draw_root_score_id_pf(lowpt,egamma,eta_upper,pt_lower)
    #c.RedrawAxis()
    
    # Legend
    graphs = c.GetListOfPrimitives()
    xmin = 0.15
    ymin = 0.4
    legend = r.TLegend(xmin,ymin,xmin+0.3,ymin+(len(graphs)+1)*0.07)
    legend.SetTextFont(42)
    legend.SetTextSize(0.045)

    temp = r.TGraph(); temp.SetMarkerColor(r.kWhite)
    text = f"pT > {pt_lower:.1f} GeV" if pt_upper is None else f"{pt_lower:.1f} < pT < {pt_upper:.1f} GeV"
    text = text.replace("pT","p_{T}")
    legend.AddEntry(temp,text,"p")
    
    for i,gr in enumerate(graphs):
        legend.AddEntry(gr.GetName(),gr.GetName(),'l')
    legend.Draw("same")
    
    # Labels and save
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='')
    c.SaveAs(f"output/roc_root_score_id_pf.pdf")
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='Preliminary')
    c.SaveAs(f"output/roc_root_score_id_pf_prelim.pdf")
