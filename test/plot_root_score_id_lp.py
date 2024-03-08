import numpy as np
import ROOT as r
from setTDRStyle import setTDRStyle
from sklearn.metrics import roc_curve, roc_auc_score
from plot_root_common import *

################################################################################

def draw_root_score_id_lp(
    LP,        # lowpt
    EG,        # egamma
    eta_u,     # upper
    pt_l,      # lower
    pt_u=None, # upper
    add_auc=True,
    **kwargs,
    ):
    
    # Labels for low-pT electrons
    has_gen =  LP.is_e     & (np.abs(LP.gen_eta)<eta_u) & (LP.gen_pt>pt_l) & (pt_u is None or LP.gen_pt<pt_u)
    has_trk = (LP.has_trk) & (np.abs(LP.trk_eta)<eta_u) & (LP.trk_pt>pt_l) & (pt_u is None or LP.trk_pt<pt_u)
    has_gsf = (LP.has_gsf) & (np.abs(LP.gsf_eta)<eta_u) & (LP.gsf_pt>pt_l) & (pt_u is None or LP.gsf_pt<pt_u)
    has_ele = (LP.has_ele) & (np.abs(LP.ele_eta)<eta_u) & (LP.ele_pt>pt_l) & (pt_u is None or LP.ele_pt<pt_u)

    branch = 'ele_mva_value_depth10'
    has_obj = has_ele
    s = LP[branch][has_gen&has_obj]
    b = LP[branch][~LP.is_e&has_obj]

    nbin=60
    xmin=-40.
    xmax=20.
    xtitle='Low-p_{T} electron ID, BDT score'
    xunit=None
    style=1
    width=2
    
    draw_root_score(
        s,
        title='Signal',
        nbin=nbin,xmin=xmin,xmax=xmax,
        xtitle=xtitle,xunit=xunit,
        style=style,width=width,color=r.kGreen+1,
        )

    draw_root_score(
        b,
        title='Background',
        nbin=nbin,xmin=xmin,xmax=xmax,
        xtitle=xtitle,xunit=xunit,
        style=style,width=width,color=r.kRed,
        same=True
        )

################################################################################

def plot_root_score_id_lp(lowpt,egamma,eta_upper,pt_lower,pt_upper=None):

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
    draw_root_score_id_lp(lowpt,egamma,eta_upper,pt_lower)
    #c.RedrawAxis()
    
    # Legend
    graphs = c.GetListOfPrimitives() 
    xmax = 0.9
    ymax = 0.9
    legend = r.TLegend(xmax-0.4,ymax-len(graphs)*0.05,xmax,ymax)
    legend.SetNColumns(2)
    legend.SetTextFont(42)
    legend.SetTextSize(0.04)

    temp = r.TGraph(); temp.SetMarkerColor(r.kWhite)
    text = f"pT > {pt_lower:.1f} GeV" if pt_upper is None else f"{pt_lower:.1f} < pT < {pt_upper:.1f} GeV"
    text = text.replace("pT","p_{T}")
    legend.AddEntry(temp,text,"p")
    
    for i,gr in enumerate(graphs):
        legend.AddEntry(gr.GetName(),gr.GetName(),'l')
    legend.Draw("same")
    
    # Labels and save
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='')
    c.SaveAs(f"output/roc_root_score_id_lp.pdf")
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='Preliminary')
    c.SaveAs(f"output/roc_root_score_id_lp_prelim.pdf")
