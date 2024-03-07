import numpy as np
import ROOT as r
from setTDRStyle import setTDRStyle
from sklearn.metrics import roc_curve, roc_auc_score
from plot_root_common import *
from plot_root_roc_id_lp import *
    
################################################################################

def plot_root_roc_id_lp_pt_binned(lowpt,egamma,eta_upper,pt_lower_v,pt_upper_v=None):

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
    gr_chance.SetName("By chance (AUC = 0.5)")
    gr_chance.SetTitle("")
    c.SetLogx()
    gr_chance.GetXaxis().SetLimits(xmin,1.)
    gr_chance.GetXaxis().SetTitle("Misidentification probability")
    gr_chance.GetYaxis().SetRangeUser(0.,1.)
    gr_chance.GetYaxis().SetNdivisions(505)
    gr_chance.GetYaxis().SetTitle("Efficiency")

    # Draw working points and ROCs (via TGraphs)
    for idx,(pt_lower,pt_upper) in enumerate(zip(pt_lower_v,pt_upper_v)):
        label = f"pT > {pt_lower} GeV" if pt_upper is None else f"{pt_lower} < pT < {pt_upper} GeV"
        draw_root_roc_id_lp(
            lowpt,
            egamma,
            eta_upper,
            pt_lower,
            pt_upper,
            label=label,
            style=[1,5,2][idx])
    
    # Legend
    graphs = c.GetListOfPrimitives()
    xmin = 0.15
    ymax = 0.8
    legend = r.TLegend(xmin,ymax-(len(graphs)+1)*0.045,xmin+0.3,ymax)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)

    # Add "By chance" first 
    legend.AddEntry(gr_chance.GetName(),gr_chance.GetName(),"l")
    
    # Custom legend entries
    tmp = r.TGraph(); tmp.SetLineColor(r.kWhite); txt = "ID, 2020Sept15"
    legend.AddEntry(tmp,txt,"l")
    
    for gr in graphs[1:]: # Ignore "By chance" in the list
        legend.AddEntry(gr.GetName(),gr.GetName(),'l')
    legend.Draw("same")
    
    # Labels and save
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='')
    c.SaveAs(f"output/plot_root_roc_id_lp_pt_binned.pdf")
    cmsLabels(c,lumiText='2018 (13 TeV)',extraText='Preliminary')
    c.SaveAs(f"output/plot_root_roc_id_lp_pt_binned_prelim.pdf")
