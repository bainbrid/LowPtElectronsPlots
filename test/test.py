from config import *
from utility import *
from parse import *
from preprocess import *

################################################################################

if __name__ == "__main__":

    data = get_data(files,nevents)
    data = preprocess(data)
    lowpt,egamma = split_by_ele_type(data)
    lowpt  = filter_data(lowpt, eta_upper=2.5,pt_lower=0.5,pt_upper=None)
    egamma = filter_data(egamma,eta_upper=2.5,pt_lower=0.5,pt_upper=None)

    eta_upper = 2.5
    pt_lower = 2.0
    pt_upper = None

    # Matplotlib version with all WPs and curves
    from plot_mpl_roc_all import *
    plot_mpl_roc_all(lowpt,egamma, eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    # ROOT version with all WPs and curves
    from plot_root_roc_all import *
    plot_root_roc_all(lowpt,egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    # ROOT version, comparison of seeds
    from plot_root_roc_seeds_comparison import *
    plot_root_roc_seeds_comparison(lowpt,egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    # ROOT version, low-pT seed ROCs
    from plot_root_roc_seed import *
    plot_root_roc_seed(lowpt,egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    # ROOT version, low-pT seed ROCs, pT-binned
    excl=True
    if excl==True: # Only one of these at a time! (Rerun for the other, as the pdf filename is the same)
        pt_lower_v = [2,1,0.5]
        pt_upper_v = [None,2,1]
        from plot_root_roc_seeds_pt_binned import *
        plot_root_roc_seeds_pt_binned(lowpt,egamma,eta_upper=eta_upper,pt_lower_v=pt_lower_v,pt_upper_v=pt_upper_v)
    else:
        pt_lower_v = [2,1,0.5]
        pt_upper_v = [None,None,None]
        from plot_root_roc_seeds_pt_binned import *
        plot_root_roc_seeds_pt_binned(lowpt,egamma,eta_upper=eta_upper,pt_lower_v=pt_lower_v,pt_upper_v=pt_upper_v)

    # ROOT version, comparison of IDs
    from plot_root_roc_ids_comparison import *
    plot_root_roc_ids_comparison(lowpt,egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    # ROOT version, PF IDs
    from plot_root_roc_id_pf import *
    plot_root_roc_id_pf(lowpt,egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    # ROOT version, low-pT IDs
    from plot_root_roc_id_lp import *
    plot_root_roc_id_lp(lowpt,egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    # ROOT version, low-pT ID ROCs, pT-binned
    pt_lower_v = [2,1,0.5]
    pt_upper_v = [None,2,1]
    from plot_root_roc_id_lp_pt_binned import *
    plot_root_roc_id_lp_pt_binned(lowpt,egamma,eta_upper=eta_upper,pt_lower_v=pt_lower_v,pt_upper_v=pt_upper_v)

    
    
