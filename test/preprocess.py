import pandas as pd
import numpy as np
import uproot
from config import *

################################################################################

def preprocess(data):

    # Data types 
    data = data.astype({'is_mc':'bool'}) # VERY IMPORTANT AS uint32 by default!
    
    if verbosity > 0:
        print(data.columns)
        for key,val in dict(data.dtypes).items(): print(f"col={key:32s} dtype={val}")

    # Filter based on tag muon pT and eta
    tag_muon_pt = 7.0
    tag_muon_eta = 1.5
    in_acc = (data.tag_pt>tag_muon_pt) & (np.abs(data.tag_eta)<tag_muon_eta)

    if verbosity > 0:
        print("Tag-side muon req, pT threshold:   ",tag_muon_pt)
        print("Tag-side muon req, eta threshold:  ",tag_muon_eta)
        print("Pre  tag-side muon req, data.shape:",data.shape)

    # Truth table when using BOTH data and MC
    # in_acc,is_mc,xor,label
    #      0     0   0    1 (data, keep)
    #      1     0   1    1 (data, keep)
    #      0     1   1    0 (MC, drop ... or keep if ONLY MC sample)
    #      1     1   0    1 (MC, keep)
        
    only_mc  = np.all(data.is_mc)
    is_mc    = data.is_mc
    trg_reqs = in_acc if only_mc else ~(is_mc & ~in_acc) # implement truth table
    data = data[trg_reqs]

    # If using data and MC, "MC == bkgd" and "data == bkgd"
    if only_mc == False: data = data[(data.is_e == data.is_mc)]
                               
    if verbosity > 0:
        print("TABLE WEIGHTS FOR MC")
        print(pd.crosstab(
            data[data['is_mc']==True]['is_e'],
            abs(data[data['is_mc']==True]['weight']-1.)<1.e-9,
            rownames=['is_e'],
            colnames=['weight==1'],
            margins=True))
        print("TABLE WEIGHTS FOR DATA")
        print(pd.crosstab(
            data[data['is_mc']==False]['is_e'],
            abs(data[data['is_mc']==False]['weight']-1.)<1.e-9,
            rownames=['is_e'],
            colnames=['weight==1'],
            margins=True))
    
    if verbosity > 0:
        print("Post tag-side muon req, data.shape:",data.shape)
            
    return data

################################################################################

def split_by_ele_type(data):

    egamma = data[data.is_egamma]           # EGamma electrons
    lowpt = data[np.invert(data.is_egamma)] # low pT electrons

    if verbosity > 0:
        print("total.shape",data.shape)
        print("egamma.shape",egamma.shape)
        print("lowpt.shape",lowpt.shape)
        pd.options.display.max_columns=None
        pd.options.display.width=None
        print(lowpt.describe().T)
        print(lowpt.info())

    return lowpt,egamma

################################################################################

def train_test_split(data, div, thr):
   mask = data.evt % div
   mask = mask < thr
   return data[mask], data[np.invert(mask)]

# Example usage
# 98% in "test" sample
#temp, test = train_test_split(lowpt, 100, 2) # _, 98%
#train, validation = train_test_split(temp, 100, 1) # 1%, 1%

################################################################################

def filter_data(
    data,
    eta_upper=2.5,
    pt_lower=2.,
    pt_upper=None,
    ):
    
    mask = np.abs(data.trk_eta) < eta_upper
    if pt_lower is not None: mask &= (data.trk_pt > pt_lower)
    if pt_upper is not None: mask &= (data.trk_pt > pt_upper)
    data = data[mask]

    return data
