from config import *
from utility import *
from parse import *
from preprocess import *

################################################################################

if __name__ == "__main__":

    data = get_data(files,nevents)
    data = preprocess(data)
    lowpt,egamma = split_by_ele_type(data)

    eta_upper = 2.5
    pt_lower = 2.
    pt_upper = None
    
#    lowpt  = filter_data(lowpt, eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)
#    egamma = filter_data(egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)
#
#    from plot_mpl_all import *
#    plot_mpl_all(lowpt,egamma, eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)
#
#    from plot_root_all import *
#    plot_root_all(lowpt,egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)
#
#    from plot_root_seeds_comparison import *
#    plot_root_seeds_comparison(lowpt,egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    from plot_root_seed import *
    plot_root_seed(lowpt,egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    excl=True
    if excl==True: # Only one of these at a time! (Rerun for the other, as the pdf filename is the same)
        pt_lower_v = [2,1,0.5]
        pt_upper_v = [None,2,1]
        from plot_root_seeds_pt_binned import *
        plot_root_seeds_pt_binned(lowpt,egamma,eta_upper=eta_upper,pt_lower_v=pt_lower_v,pt_upper_v=pt_upper_v)
    else:
        pt_lower_v = [2,1,0.5]
        pt_upper_v = [None,None,None]
        from plot_root_seeds_pt_binned import *
        plot_root_seeds_pt_binned(lowpt,egamma,eta_upper=eta_upper,pt_lower_v=pt_lower_v,pt_upper_v=pt_upper_v)

    
    
