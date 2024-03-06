import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tdrstyle.mplstyle')
from sklearn.metrics import roc_curve, roc_auc_score
from plot_mpl_common import *

################################################################################

def draw_mpl_roc_all(lowpt,egamma,eta_upper,pt_lower,**kwargs):

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
    #draw_mpl_wp(trk_eff,trk_fr,label='Low-pT track',mfc='black')
    
    # Eff and fake rate for low-pT GSF tracks
    denom = has_gen&has_trk; numer = has_gsf&denom;
    gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    denom = has_trk&(~lowpt.is_e); numer = has_gsf&denom;
    gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    draw_mpl_wp(gsf_eff,gsf_fr,label='Low-pT GSF track',mfc='blue',markersize=10)
    
    # Eff and fake rate for low-pT electrons
    denom = has_gen&has_trk; numer = has_ele&denom;
    ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    denom = has_trk&(~lowpt.is_e); numer = has_ele&denom;
    ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    draw_mpl_wp(ele_eff,ele_fr,label='Low-pT electron',mfc='red')
    
    # ROC for low-pT unbiased seed
    branch = 'gsf_bdtout1'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(lowpt.is_e[has_obj],lowpt[branch][has_obj])
    auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
    draw_mpl_roc(tpr,fpr,auc,ele_eff,ele_fr,label='Unbiased seed',color='blue')

    # ROC for low-pT biased seed
    branch = 'gsf_bdtout2'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(lowpt.is_e[has_obj],lowpt[branch][has_obj])
    auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
    draw_mpl_roc(tpr,fpr,auc,ele_eff,ele_fr,label='Biased seed',color='blue',linestyle='dashed')

    # ID ROC for 2020Sept15
    branch = 'ele_mva_value_depth10'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(lowpt.is_e[has_obj],lowpt[branch][has_obj])
    auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
    draw_mpl_roc(tpr,fpr,auc,ele_eff,ele_fr,label='ID, 2020Sept15',color='red')

    # ID ROC for 2019Aug07
    branch = 'ele_mva_value'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(lowpt.is_e[has_obj],lowpt[branch][has_obj])
    auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
    draw_mpl_roc(tpr,fpr,auc,ele_eff,ele_fr,label='ID, 2019Aug07',color='red',linestyle='dashdot')

    # ID ROC for 2021May17
    branch = 'ele_mva_value_depth13'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(lowpt.is_e[has_obj],lowpt[branch][has_obj])
    auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
    draw_mpl_roc(tpr,fpr,auc,ele_eff,ele_fr,label='ID, 2021May17',color='red',linestyle='dashed')

    # ID ROC for 2020Nov28
    branch = 'ele_mva_value_depth11'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(lowpt.is_e[has_obj],lowpt[branch][has_obj])
    auc = roc_auc_score(lowpt.is_e[has_obj],lowpt[branch][has_obj]) if len(set(lowpt.is_e[has_obj])) > 1 else 0.
    draw_mpl_roc(tpr,fpr,auc,ele_eff,ele_fr,label='ID, 2020Nov28',color='red',linestyle='dotted')
    
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
    #draw_mpl_wp(trk_eff,trk_fr,label='PF track',mfc='black')
    
    # Eff and fake rate for EGamma seeds
    denom = has_gen&has_trk; numer = has_gsf&denom;
    gsf_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    denom = has_trk&(~egamma.is_e); numer = has_gsf&denom;
    gsf_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    draw_mpl_wp(gsf_eff,gsf_fr,label='Egamma seed',mfc='orange',markersize=10)
    
    # Eff and fake rate for PF electrons
    denom = has_gen&has_trk; numer = has_ele&denom;
    ele_eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    denom = has_trk&(~egamma.is_e); numer = has_ele&denom;
    ele_fr = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
    draw_mpl_wp(ele_eff,ele_fr,label='PF electron',mfc='purple')

    # ID ROC for PF (default)
    branch = 'ele_mva_value'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(egamma.is_e[has_obj],egamma[branch][has_obj])
    auc = roc_auc_score(egamma.is_e[has_obj],egamma[branch][has_obj]) if len(set(egamma.is_e[has_obj])) > 1 else 0.
    draw_mpl_roc(tpr,fpr,auc,ele_eff,ele_fr,label='ID, PF',color='purple',linestyle='dotted')

    # ID ROC for PF (retrained)
    branch = 'ele_mva_value_retrained'
    has_obj = has_ele
    fpr,tpr,thr = roc_curve(egamma.is_e[has_obj],egamma[branch][has_obj])
    auc = roc_auc_score(egamma.is_e[has_obj],egamma[branch][has_obj]) if len(set(egamma.is_e[has_obj])) > 1 else 0.
    draw_mpl_roc(tpr,fpr,auc,ele_eff,ele_fr,label='ID, PF retrained',color='purple')

################################################################################

def plot_mpl_roc_all(lowpt,egamma,eta_upper,pt_lower,pt_upper=None):

    # Figure 
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111)
    plt.title('Electron performance')

    # Axes
    pt_threshold = f"pT > {pt_lower:.1f} GeV" if pt_upper is None else f"{pt_lower:.1f} < pT < {pt_upper:.1f} GeV"
    plt.xlabel(f'Mistag rate (w.r.t. KF tracks, {pt_threshold}, |eta| < {eta_upper})')
    plt.ylabel(f'Efficiency (w.r.t. KF tracks, {pt_threshold}, |eta| < {eta_upper})')
    plt.xlim(1.e-4,1.)
    plt.ylim([0., 1.])
    plt.gca().set_xscale('log')
    plt.grid(True)
    ax.tick_params(axis='x', pad=10.)

    # By chance
    plt.plot(
        np.arange(0.,1.,plt.xlim()[0]),
        np.arange(0.,1.,plt.xlim()[0]),
        ls='dotted',lw=0.5,
        label="By chance (AUC=0.5)")

    # Draw working points and ROCs
    draw_mpl_roc_all(lowpt,egamma,eta_upper,pt_lower)

    # Finish up
    plt.legend(loc='lower right',facecolor='white',framealpha=None,frameon=False)
    plt.tight_layout()
    name = 'plots/roc_mpl_all.pdf'
    print(f'Saving {name}')
    plt.savefig(name)
    plt.clf()
    plt.close()
