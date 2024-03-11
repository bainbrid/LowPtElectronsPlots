plot_root_vars_id_pf.py import numpy as np
import ROOT as r
from setTDRStyle import setTDRStyle
from sklearn.metrics import roc_curve, roc_auc_score
from plot_root_common import *

################################################################################

def draw_root_var_id_pf(
    LP,        # lowpt
    EG,        # egamma
    eta_u,     # upper
    pt_l,      # lower
    pt_u=None, # upper
    **kwargs,
    ):
    
    # Labels for low-pT electrons
    has_gen =  EG.is_e       & (np.abs(EG.gen_eta)<eta_u)   & (EG.gen_pt>pt_l)   & (pt_u is None or EG.gen_pt<pt_u)  
    has_trk = (EG.has_trk)   & (np.abs(EG.trk_eta)<eta_u)   & (EG.trk_pt>pt_l)   & (pt_u is None or EG.trk_pt<pt_u)  
    has_gsf = (EG.has_pfgsf) & (np.abs(EG.pfgsf_eta)<eta_u) & (EG.pfgsf_pt>pt_l) & (pt_u is None or EG.pfgsf_pt<pt_u)
    has_ele = (EG.has_ele)   & (np.abs(EG.ele_eta)<eta_u)   & (EG.ele_pt>pt_l)   & (pt_u is None or EG.ele_pt<pt_u)  

    feature = kwargs.get('feature','Unknown feature')
    branch = feature
    has_obj = has_ele
    s = EG[branch][has_gen&has_obj]
    b = EG[branch][~EG.is_e&has_obj]
    
    draw_root_var(
        s,
        name=f"{feature}_S",
        title='B^{+} #rightarrow K^{+}e^{+}e^{#minus}',
        nbin=kwargs.get('nbin',None),
        xmin=kwargs.get('xmin',None),
        xmax=kwargs.get('xmax',None),
        ymin=kwargs.get('ymin',None),
        ymax=kwargs.get('ymax',None),
        xtitle=kwargs.get('xtitle',None),
        xunit=kwargs.get('xunit',None),
        style=1,
        width=2,
        color=r.kGreen+1,
        )

    draw_root_var(
        b,
        is_data=True,
        name=f"{feature}_B",
        title='Control data',
        nbin=kwargs.get('nbin',None),
        xmin=kwargs.get('xmin',None),
        xmax=kwargs.get('xmax',None),
        ymin=kwargs.get('ymin',None),
        ymax=kwargs.get('ymax',None),
        xtitle=kwargs.get('xtitle',None),
        xunit=kwargs.get('xunit',None),
        style=1,
        width=2,
        color=r.kRed,
        same=True
        )

################################################################################

def plot_root_var(lowpt,egamma,eta_upper,pt_lower,pt_upper=None,**kwargs):

    feature = kwargs.get("feature","Unknown feature")
    
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
    draw_root_var_id_pf(lowpt,egamma,eta_upper,pt_lower,**kwargs)
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
    #cmsLabels(c,lumiText='2018 (13 TeV)',extraText='')
    #c.SaveAs(f"output/vars/roc_root_{feature}_id_pf.pdf")
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='Preliminary')
    c.SaveAs(f"output/vars/roc_root_{feature}_id_pf_prelim.pdf")

    
################################################################################

def plot_root_vars_id_pf(lowpt,egamma,eta_upper,pt_lower,pt_upper=None):

    features = [
        # BDT scores
        {'feature':'ele_mva_value','xtitle':'PF electron ID, BDT score','xunit':None,'nbin':60,'xmin':-15.,'xmax':15.,'ymin':0.,'ymax':0.2,},
        {'feature':'ele_mva_value_retrained','xtitle':'PF electron retrained ID, BDT score','xunit':None,'nbin':60,'xmin':-15.,'xmax':15.,'ymin':0.,'ymax':0.2,},
    ]

    for kwargs in features:
        plot_root_var(lowpt,egamma,eta_upper,pt_lower,pt_upper=None,**kwargs)
