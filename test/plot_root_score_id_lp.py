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

    nbin=80
    xmin=-20.
    xmax=20.
    ymin=0.
    ymax=0.12
    xtitle='Low-p_{T} electron ID, BDT score'
    xunit=None
    style=1
    width=2

    draw_root_var(
        s,
        name=f'{__name__}_Signal',
        title='Signal',
        #title='B^{+} #rightarrow K^{+}e^{+}e^{#minus}',
        nbin=nbin,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        xtitle=xtitle,
        xunit=xunit,
        style=style,
        width=width,
        color=r.kGreen+1,
        )

    draw_root_var(
        b,
        #is_data=True,
        name=f'{__name__}_Background',
        title='Background',
        #title='Control data',
        nbin=nbin,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        xtitle=xtitle,
        xunit=xunit,
        style=style,
        width=width,
        color=r.kRed,
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
    xmax = 0.92; ymax = 0.88
    length = 0.2; height = 0.055; size = 0.045
    graphs = c.GetListOfPrimitives()
    legend = r.TLegend(xmax-1.2*length,ymax-len(graphs)*height,xmax,ymax)
    #legend.SetNColumns(2)
    legend.SetTextFont(42)
    legend.SetTextSize(size)
    for i,gr in enumerate(graphs):
        legend.AddEntry(gr.GetName(),gr.GetTitle(),'l' if 'data' not in gr.GetTitle() else 'pe')
    legend.Draw("same")

    # Text
    text = f"pT > {pt_lower:.1f} GeV" if pt_upper is None else f"{pt_lower:.1f} < pT < {pt_upper:.1f} GeV"
    text = text.replace("pT","p_{T}")
    txt = r.TLatex(0.45,0.83,text)
    txt.SetNDC(True)
    txt.SetTextAlign(11)
    txt.SetTextFont(42)
    txt.SetTextSize(size)
    txt.Draw("same")
    
    
    # Labels and save
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='')
    c.SaveAs(f"output/roc_root_score_id_lp.pdf")
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='Preliminary')
    c.SaveAs(f"output/roc_root_score_id_lp_prelim.pdf")
