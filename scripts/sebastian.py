################################################################################
# Imports ...

from __future__ import print_function
import builtins
import future
from future.utils import raise_with_traceback

import uproot
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=False)
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties

import ROOT as r 
from setTDRStyle import setTDRStyle

################################################################################
# I/O

# Input files
#filenames = ["../data/170823/nonres_large/output_0.root"] # small
filenames = [
    #"../data/170823/nonres_large/output_000.root", # large
    #"output/output_old.root",
    "output/output.root",
    ]
print('Input filenames:')
for idx,filename in enumerate(filenames):
    print(f' #{idx}: {filename}')
    
columns = [
    # LABELING
    'is_e','is_egamma',
    'has_trk','has_seed','has_gsf','has_pfgsf','has_ele',
    # KINE
    'gen_pt','gen_eta',
    'tag_pt','tag_eta',
    'trk_pt','trk_eta',
    'gsf_pt','gsf_eta',
    'pfgsf_pt','pfgsf_eta',
    'ele_pt','ele_eta',
    # SCORES
    'gsf_bdtout1', # "pT-biased seeding BDT" score
    'gsf_bdtout2', # "Unbiased seeding BDT" score
    'ele_mva_value', # Low-pT electron ID score
    'ele_mva_value_depth10',
    'ele_mva_value_depth11',
    'ele_mva_value_depth13',
    'ele_mva_value_retrained', # Retrained PF electron ID score
    # MISC
    'evt','weight','rho',
    ]
columns = list(set(columns))

# Extract branches from root file as a pandas data frame
df = [ uproot.open(i)['ntuplizer/tree'].arrays(columns,library="pd")  for i in filenames ]
data = pd.concat(df)

################################################################################
# Filters applied to branches
################################################################################

# Filter data based on tag-side muon pT and eta
tag_muon_pt = 7.0
tag_muon_eta = 1.5
data = data[ (data.tag_pt>tag_muon_pt) & (np.abs(data.tag_eta)<tag_muon_eta) ]
print(data.describe(include='all').T)

# Split into low-pT and PF parts
lowpt = data[np.invert(data.is_egamma)] # low pT electrons
egamma = data[data.is_egamma]           # EGamma electrons

pt_lower = 2.0
pt_upper = 1.e6 #None # or 5.0 ?

mask = (np.abs(lowpt.trk_eta) < 2.5) & (lowpt.trk_pt>pt_lower) & (lowpt.trk_pt<pt_upper)
lowpt = lowpt[mask]

mask = (np.abs(egamma.trk_eta)<2.5) & (egamma.trk_pt>pt_lower) & (egamma.trk_pt<pt_upper)
egamma = egamma[mask]

################################################################################
# ROC curve using matplotlib
################################################################################

loose_wp_thr = -2.780
tight_wp_thr = 5.076

plt.figure(figsize=(6,6))
ax = plt.subplot(111)
plt.title('Low-pT electron performance (BParking)')

xmin = 1.e-4
plt.xlim(xmin,1.)
plt.ylim([0., 1.])

plt.xlabel(f'Mistag rate (w.r.t. KF tracks, pT > {pt_lower:.1f} GeV)')
plt.ylabel(f'Efficiency (w.r.t. KF tracks, pT > {pt_lower:.1f} GeV)')
ax.tick_params(axis='x', pad=10.)
plt.gca().set_xscale('log')
plt.grid(True)

# "by chance" line
plt.plot(np.arange(0.,1.,plt.xlim()[0]),np.arange(0.,1.,plt.xlim()[0]),ls='dotted',lw=0.5,label="By chance")

has_gen =  lowpt.is_e     & (lowpt.gen_pt>pt_lower) & (np.abs(lowpt.gen_eta)<2.5)
has_trk = (lowpt.has_trk) & (lowpt.trk_pt>pt_lower) & (np.abs(lowpt.trk_eta)<2.5)
has_gsf = (lowpt.has_gsf) & (lowpt.gsf_pt>pt_lower) & (np.abs(lowpt.gsf_eta)<2.5)
has_ele = (lowpt.has_ele) & (lowpt.ele_pt>pt_lower) & (np.abs(lowpt.ele_eta)<2.5)

print(pd.crosstab(
    lowpt.is_e,
    [has_ele],
    rownames=['is_e'],
    colnames=['has_ele'],
    margins=True))

denom = has_gen; numer = has_trk&denom;
trk_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
denom = has_trk&(~lowpt.is_e); numer = has_trk&denom;
trk_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

denom = has_gen&has_trk; numer = has_gsf&denom;
gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
denom = has_trk&(~lowpt.is_e); numer = has_gsf&denom;
gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

denom = has_gen&has_trk; numer = has_ele&denom;
ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
denom = has_trk&(~lowpt.is_e); numer = has_ele&denom;
ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

plt.plot(
    [gsf_fr],
    [gsf_eff],
    marker='o', markerfacecolor='blue', markeredgecolor='blue',
    markersize=8,linestyle='none',
    label='Low-pT GSF track',
    )

plt.plot(
    [ele_fr],
    [ele_eff],
    marker='o', markerfacecolor='red', markeredgecolor='red', 
    markersize=8,linestyle='none',
    label='Low-pT electron',
    )
   
biased_branch = 'gsf_bdtout2'
has_obj = has_ele
biased_fpr,biased_tpr,biased_thr = roc_curve(lowpt.is_e[has_obj],lowpt[biased_branch][has_obj])
biased_auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[biased_branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
plt.plot(
    biased_fpr*ele_fr,
    biased_tpr*ele_eff,
    linestyle='solid', color='blue', linewidth=1.0,
    label='Biased seed (AUC={:.3f})'.format(biased_auc)
    )

unbias_branch = 'gsf_bdtout1'
has_obj = has_ele
unbias_fpr,unbias_tpr,unbias_thr = roc_curve(lowpt.is_e[has_obj],lowpt[unbias_branch][has_obj])
unbias_auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[unbias_branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
plt.plot(
    unbias_fpr*ele_fr,
    unbias_tpr*ele_eff,
    linestyle='solid', color='green', linewidth=1.0,
    label='Unbiased seed ({:.3f})'.format(unbias_auc)
    )

# 2020Sept15
id_2020Sept15_branch = 'ele_mva_value_depth10'
has_obj = has_ele
id_2020Sept15_fpr,id_2020Sept15_tpr,id_2020Sept15_thr = roc_curve(lowpt.is_e[has_obj],lowpt[id_2020Sept15_branch][has_obj])
id_2020Sept15_auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[id_2020Sept15_branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
plt.plot(
    id_2020Sept15_fpr*ele_fr, 
    id_2020Sept15_tpr*ele_eff,
    linestyle='solid', color='red', linewidth=1.0,
    label='2020Sept15 ({:.3f})'.format(id_2020Sept15_auc))

# 2020Sept15 TEST
id_2020Sept15_test_branch = 'ele_mva_value_depth10'
has_obj = has_trk
id_2020Sept15_test_fpr,id_2020Sept15_test_tpr,id_2020Sept15_test_thr = roc_curve(lowpt.is_e[has_obj],lowpt[id_2020Sept15_test_branch][has_obj])
id_2020Sept15_test_auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[id_2020Sept15_test_branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
plt.plot(
    id_2020Sept15_test_fpr,
    id_2020Sept15_test_tpr,
    linestyle='dashed', color='pink', linewidth=1.0,
    label='2020Sept15 TEST ({:.3f})'.format(id_2020Sept15_test_auc))

# PF electron
has_pfgen   =  egamma.is_e       & (egamma.gen_pt>pt_lower)   & (np.abs(egamma.gen_eta)<2.5)
has_pftrk   = (egamma.has_trk)   & (egamma.trk_pt>pt_lower)   & (np.abs(egamma.trk_eta)<2.5)
has_pfgsf   = (egamma.has_pfgsf) & (egamma.pfgsf_pt>pt_lower) & (np.abs(egamma.pfgsf_eta)<2.5)
has_pfele   = (egamma.has_ele)   & (egamma.ele_pt>pt_lower)   & (np.abs(egamma.ele_eta)<2.5)

print(pd.crosstab(
    egamma.is_e,
    [has_pfele],
    rownames=['is_e'],
    colnames=['has_pfele'],
    margins=True))

denom = has_pfgen; numer = has_pftrk&denom
pftrk_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
denom = has_pftrk&(~egamma.is_e); numer = has_pftrk&denom
pftrk_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

denom = has_pfgen&has_pftrk; numer = has_pfgsf&denom
pfgsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
denom = has_pftrk&(~egamma.is_e); numer = has_pfgsf&denom
pfgsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

denom = has_pfgen&has_pftrk; numer = has_pfele&denom
pfele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
denom = has_pftrk&(~egamma.is_e); numer = has_pfele&denom
pfele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.

plt.plot(
    [pfgsf_fr],
    [pfgsf_eff],
    marker='o', color='orange', 
    markersize=10, linestyle='none',
    label='EGamma seed'
    )

plt.plot(
    [pfele_fr],
    [pfele_eff],
    marker='o', color='purple', 
    markersize=8, linestyle='none',
    label='PF electron'
    )

# PF ID (default)
pf_id_branch = 'ele_mva_value'
has_obj = has_pfele
pf_id_fpr,pf_id_tpr,pf_id_thr = roc_curve(egamma.is_e[has_obj],egamma[pf_id_branch][has_obj])
pf_id_auc = roc_auc_score(egamma.is_e[has_obj],egamma[pf_id_branch][has_obj]) if len(set(egamma.is_e[has_obj])) > 1 else 0.
plt.plot(
    pf_id_fpr*pfele_fr,
    pf_id_tpr*pfele_eff,
    linestyle='dotted', color='purple', linewidth=1.0,
    label='PF default ID ({:.3f})'.format(pf_id_auc)
    )

# PF ID (retrained)
pf_id_retrain_branch = 'ele_mva_value_retrained'
has_obj = has_pfele
pf_id_retrain_fpr,pf_id_retrain_tpr,pf_id_retrain_thr = roc_curve(egamma.is_e[has_obj],egamma[pf_id_retrain_branch][has_obj])
pf_id_retrain_auc = roc_auc_score(egamma.is_e[has_obj],egamma[pf_id_retrain_branch][has_obj]) if len(set(egamma.is_e[has_obj])) > 1 else 0.
plt.plot(
    pf_id_retrain_fpr*pfele_fr, 
    pf_id_retrain_tpr*pfele_eff,
    linestyle='solid', color='purple', linewidth=1.0,
    label='PF retrained ID ({:.3f})'.format(pf_id_retrain_auc)
    )

# Loose working point (same fake rate)
loose_wp_idx = np.abs(id_2020Sept15_thr-loose_wp_thr).argmin()
loose_wp_fr  = id_2020Sept15_fpr[loose_wp_idx]*ele_fr
loose_wp_eff = id_2020Sept15_tpr[loose_wp_idx]*ele_eff
loose_wp_thr = id_2020Sept15_thr[loose_wp_idx]
loose_wp = lowpt[id_2020Sept15_branch][has_ele]>loose_wp_thr

tight_wp_idx = np.abs(id_2020Sept15_thr-tight_wp_thr).argmin()
tight_wp_fr  = id_2020Sept15_fpr[tight_wp_idx]*ele_fr
tight_wp_eff = id_2020Sept15_tpr[tight_wp_idx]*ele_eff
tight_wp_thr = id_2020Sept15_thr[tight_wp_idx]
tight_wp = lowpt[id_2020Sept15_branch][has_ele]>tight_wp_thr

plt.plot(
    [loose_wp_fr],
    [loose_wp_eff],
    marker='o', color='red', 
    markersize=6, linestyle='none',
    label='Loose ID WP'
    )

plt.plot(
    [tight_wp_fr],
    [tight_wp_eff],
    marker='o', color='red', 
    markersize=6, linestyle='none',
    label='Tight ID WP')

plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
plt.tight_layout()
plt.savefig('./roc.pdf')
plt.clf()
plt.close()
print('Created file: ./roc.pdf')

################################################################################
# ROC curve using ROOT
################################################################################

setTDRStyle()
c = r.TCanvas("","",800,600)
c.SetTopMargin(0.08)
c.SetBottomMargin(0.14)
c.SetLeftMargin(0.12)
c.SetRightMargin(0.05)
c.SetLogx()
   
# Chance graph
chance_tpr = np.array(np.arange(xmin,1.,xmin))
chance_fpr = np.array(np.arange(xmin,1.,xmin))
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

# Unbiased seeding BDT 
g_unbias = r.TGraph(len(unbias_fpr), unbias_fpr*ele_fr, unbias_tpr*ele_eff)
g_unbias.SetTitle("AUC = {:.2f}".format(unbias_auc))
g_unbias.SetLineStyle(3)
g_unbias.SetLineWidth(2)
g_unbias.SetLineColor(r.kGreen+3)
#g_unbias.Draw("Lsame")

# pT-biased seeding BDT 
g_biased = r.TGraph(len(biased_fpr), biased_fpr*ele_fr, biased_tpr*ele_eff)
g_biased.SetTitle("AUC = {:.2f}".format(biased_auc))
g_biased.SetLineStyle(5)
g_biased.SetLineWidth(2)
g_biased.SetLineColor(r.kGreen+3)
#g_biased.Draw("Lsame")

# Low-pT electron ID
g_id = r.TGraph(len(id_2020Sept15_fpr), id_2020Sept15_fpr*ele_fr, id_2020Sept15_tpr*ele_eff)
g_id.SetTitle("AUC = {:.2f}".format(id_2020Sept15_auc))
g_id.SetLineStyle(1)
g_id.SetLineWidth(2)
g_id.SetLineColor(r.kBlue)
g_id.Draw("Lsame")

# Low-pT electron candidate
m_ele = r.TGraph()
m_ele.SetPoint(0,ele_fr,ele_eff)
m_ele.SetMarkerStyle(20)
m_ele.SetMarkerSize(2)
m_ele.SetMarkerColor(r.kBlue)
m_ele.Draw("Psame")

# PF electron ID
g_pf_id_old = r.TGraph(len(pf_id_fpr), pf_id_fpr*pfele_fr, pf_id_tpr*pfele_eff)
g_pf_id_old.SetTitle("AUC = {:.2f}".format(pf_id_auc))
g_pf_id_old.SetLineStyle(2)
g_pf_id_old.SetLineWidth(2)
g_pf_id_old.SetLineColor(r.kRed)
g_pf_id_old.Draw("Lsame")

# PF electron retrained ID
g_pf_id_new = r.TGraph(len(pf_id_retrain_fpr), pf_id_retrain_fpr*pfele_fr, pf_id_retrain_tpr*pfele_eff)
g_pf_id_new.SetTitle("AUC = {:.2f}".format(pf_id_retrain_auc))
g_pf_id_new.SetLineStyle(1)
g_pf_id_new.SetLineWidth(2)
g_pf_id_new.SetLineColor(r.kRed)
g_pf_id_new.Draw("Lsame")

# PF electron candidate
m_pfele = r.TGraph()
m_pfele.SetPoint(0,pfele_fr,pfele_eff)
m_pfele.SetMarkerStyle(21)
m_pfele.SetMarkerSize(2)
m_pfele.SetMarkerColor(r.kRed)
m_pfele.Draw("Psame")

legend = r.TLegend(0.45,0.2,0.8,0.2+5*0.055)
legend.SetTextFont(42)
legend.SetTextSize(0.04)
temp = r.TGraph()
temp.SetMarkerColor(r.kWhite)
legend.AddEntry(temp,"p_{T} > 2 GeV, |#eta| < 2.5","p")
legend.AddEntry(m_ele,"Low-p_{T} electron cand.","p")
legend.AddEntry(g_id,"Low-p_{T} identification","l")
legend.AddEntry(m_pfele,"PF electron","p")
legend.AddEntry(g_chance,"By chance","l")
legend.Draw("same")
g_chance.Draw("same") # over legend

# Latex box
latex = r.TLatex()
latex.SetNDC()
latex.SetTextAngle(0)
latex.SetTextColor(r.kBlack)
c.cd()
top = c.GetTopMargin()
right = c.GetRightMargin()

# CMS (Preliminary) label
cmsText = 'CMS'
cmsTextFont = 61
cmsTextSize = 0.75
latex.SetTextFont(cmsTextFont)
latex.SetTextSize(cmsTextSize*top)
latex.SetTextAlign(11)
latex.DrawLatex(0.16,0.83,cmsText)

# CMS lumi label
lumiText = '2018 (13 TeV)'
lumiTextSize = 0.6
lumiTextOffset = 0.2
latex.SetTextFont(42)
latex.SetTextAlign(31)
latex.SetTextSize(lumiTextSize*top)
latex.DrawLatex(1-right,1-top+lumiTextOffset*top,lumiText)
   
c.SaveAs(f"./roc_root.pdf")
