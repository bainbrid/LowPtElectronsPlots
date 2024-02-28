from config import *
from utility import *
from parse import *
from preprocess import *
from plot_mpl import *
    
################################################################################

if __name__ == "__main__":

    data = get_data(files,nevents)
    data = preprocess(data)
    lowpt,egamma = split_by_ele_type(data)

    eta_upper = 2.5
    pt_lower = 2.
    pt_upper = None
    
    lowpt  = filter_data(lowpt, eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)
    egamma = filter_data(egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    plot_roc_all_mpl(lowpt,egamma, eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)
    #plot_roc_all_root(lowpt,egamma,eta_upper=eta_upper,pt_lower=pt_lower,pt_upper=pt_upper)

    
