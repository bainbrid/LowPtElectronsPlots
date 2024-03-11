import numpy as np
import ROOT as r
from setTDRStyle import setTDRStyle
from sklearn.metrics import roc_curve, roc_auc_score
from plot_root_common import *

################################################################################

def draw_root_var_id_lp(
    LP,        # lowpt
    EG,        # egamma
    eta_u,     # upper
    pt_l,      # lower
    pt_u=None, # upper
    **kwargs,
    ):
    
    # Labels for low-pT electrons
    has_gen =  LP.is_e     & (np.abs(LP.gen_eta)<eta_u) & (LP.gen_pt>pt_l) & (pt_u is None or LP.gen_pt<pt_u)
    has_trk = (LP.has_trk) & (np.abs(LP.trk_eta)<eta_u) & (LP.trk_pt>pt_l) & (pt_u is None or LP.trk_pt<pt_u)
    has_gsf = (LP.has_gsf) & (np.abs(LP.gsf_eta)<eta_u) & (LP.gsf_pt>pt_l) & (pt_u is None or LP.gsf_pt<pt_u)
    has_ele = (LP.has_ele) & (np.abs(LP.ele_eta)<eta_u) & (LP.ele_pt>pt_l) & (pt_u is None or LP.ele_pt<pt_u)

    feature = kwargs.get('feature','Unknown feature')
    branch = feature
    has_obj = has_ele
    s = LP[branch][has_gen&has_obj]
    b = LP[branch][~LP.is_e&has_obj]
    
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
    draw_root_var_id_lp(lowpt,egamma,eta_upper,pt_lower,**kwargs)
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
    #c.SaveAs(f"output/vars/roc_root_{feature}_id_lp.pdf")
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='Preliminary')
    c.SaveAs(f"output/vars/roc_root_{feature}_id_lp_prelim.pdf")

    
################################################################################

def plot_root_vars_id_lp(lowpt,egamma,eta_upper,pt_lower,pt_upper=None):

    features = [
        # BDT scores
        {'feature':'ele_mva_value_depth10','xtitle':'Electron ID, BDT score','xunit':None,'nbin':60,'xmin':-15.,'xmax':15.,'ymin':0.,'ymax':0.12,},
        {'feature':'gsf_bdtout1','xtitle':'Unbiased seed BDT score','xunit':None,'nbin':60,'xmin':-15.,'xmax':15.,'ymin':0.,'ymax':0.12,},
        {'feature':'gsf_bdtout2','xtitle':'Biased seed BDT score','xunit':None,'nbin':60,'xmin':-15.,'xmax':15.,'ymin':0.,'ymax':0.2,},
        # Kinematic features
        {'feature':'eid_ele_pt','xtitle':'Electron p_{T}','xunit':'GeV','nbin':100,'xmin':0.,'xmax':20.,'ymin':0.,'ymax':0.15,},
        {'feature':'eid_trk_p','xtitle':'Electron track momentum','xunit':'GeV','nbin':100,'xmin':0.,'xmax':20.,'ymin':0.,'ymax':0.15,},
        {'feature':'eid_brem_frac','xtitle':'(p_{T}^{in} - p_{T}^{out}) / p_{T}^{in}','xunit':None,'nbin':100,'xmin':0.,'xmax':1.,'ymin':0.,'ymax':0.2,},
        # Track related 
        {'feature':'eid_trk_chi2red','xtitle':'Track #chi^{2}/dof','xunit':None,'nbin':100,'xmin':0.,'xmax':10.,'ymin':0.,'ymax':0.15,},
        {'feature':'eid_trk_nhits','xtitle':'Track number of hits','xunit':None,'nbin':31,'xmin':-0.5,'xmax':30.5,'ymin':0.,'ymax':0.15,},
        {'feature':'eid_gsf_chi2red','xtitle':'GSF track #chi^{2}/dof','xunit':None,'nbin':100,'xmin':0.,'xmax':10.,'ymin':0.,'ymax':0.15,},
        {'feature':'eid_gsf_nhits','xtitle':'GSF track number of hits','xunit':None,'nbin':31,'xmin':-0.5,'xmax':30.5,'ymin':0.,'ymax':0.25,},
        # SuperCluster related 
        {'feature':'eid_sc_E','xtitle':'Super cluster energy','xunit':'GeV','nbin':100,'xmin':0.,'xmax':20.,'ymin':0.,'ymax':0.1,},
        {'feature':'eid_sc_eta','xtitle':'Super cluster #eta','xunit':None,'nbin':104,'xmin':-2.6,'xmax':2.6,'ymin':0.,'ymax':0.025,},
        {'feature':'eid_sc_etaWidth','xtitle':'Super cluster #Delta#eta','xunit':None,'nbin':100,'xmin':-0,'xmax':0.5,'ymin':0.,'ymax':0.2,},
        {'feature':'eid_sc_phiWidth','xtitle':'Super cluster #Delta#phi','xunit':None,'nbin':100,'xmin':0.,'xmax':0.5,'ymin':0.,'ymax':0.2,},
        # Shape variables
        {'feature':'eid_shape_full5x5_sigmaIetaIeta','xtitle':'#sigma_{i#etai#eta}','xunit':None,'nbin':100,'xmin':0.,'xmax':0.1,'ymin':0.,'ymax':0.2,},
        {'feature':'eid_shape_full5x5_sigmaIphiIphi','xtitle':'#sigma_{i#phii#phi}','xunit':None,'nbin':100,'xmin':0.,'xmax':0.1,'ymin':0.,'ymax':0.2,},
        {'feature':'eid_shape_full5x5_circularity','xtitle':'Circularity','xunit':None,'nbin':100,'xmin':0.,'xmax':1.,'ymin':0.,'ymax':0.2,},
        {'feature':'eid_shape_full5x5_r9','xtitle':'R_{9} (5#times5)','xunit':None,'nbin':100,'xmin':0.,'xmax':2.,'ymin':0.,'ymax':0.12,},
        {'feature':'eid_shape_full5x5_HoverE','xtitle':'H/E (5#times5)','xunit':None,'nbin':100,'xmin':0.,'xmax':2.,'ymin':0.,'ymax':0.25,},
        # Track-cluster matching
        {'feature':'eid_match_SC_EoverP','xtitle':'Super cluster E/p','xunit':None,'nbin':100,'xmin':0.,'xmax':2.,'ymin':0.,'ymax':0.2,},
        {'feature':'eid_match_SC_dEta','xtitle':'#Delta#eta(track, super cluster)','xunit':None,'nbin':100,'xmin':-0.5,'xmax':0.5,'ymin':0.,'ymax':0.25,},
        {'feature':'eid_match_SC_dPhi','xtitle':'#Delta#phi(track, super cluster)','xunit':None,'nbin':100,'xmin':-0.5,'xmax':0.5,'ymin':0.,'ymax':0.25,},
        {'feature':'eid_match_eclu_EoverP','xtitle':'Seed cluster E/p','xunit':None,'nbin':100,'xmin':0.,'xmax':1.,'ymin':0.,'ymax':0.06,},
        {'feature':'eid_match_seed_dEta','xtitle':'#Delta#eta(track, seed cluster)','xunit':None,'nbin':100,'xmin':-0.5,'xmax':0.5,'ymin':0.,'ymax':0.25,},
        # Misc
        {'feature':'eid_rho','xtitle':'#rho','xunit':None,'nbin':100,'xmin':0.,'xmax':50.,'ymin':0.,'ymax':0.1,},
    ]

    for kwargs in features:
        plot_root_var(lowpt,egamma,eta_upper,pt_lower,pt_upper=None,**kwargs)
