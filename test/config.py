from argparse import ArgumentParser

################################################################################

parser = ArgumentParser()
parser.add_argument('--verbosity',default=0,type=int)
parser.add_argument('--nevents',default=-1,type=int)
parser.add_argument('--sample',default="large",type=str)
args = parser.parse_args()
print("Command line args:",vars(args))

verbosity = args.verbosity
nevents = args.nevents

################################################################################

files = {
    "small":["./data/output_small.root","./data/output_data_small.root"],
    "medium":["./data/output_medium.root","./data/output_data_medium.root"],
    "large":["./data/output_large.root","./data/output_data_large.root"],
    }.get(args.sample,["./data/output_small.root","./data/output_data_small.root"])

################################################################################

features = [ # ORDER IS VERY IMPORTANT ! 
   'eid_rho',
   'eid_ele_pt',
   'eid_sc_eta',
   'eid_shape_full5x5_sigmaIetaIeta',
   'eid_shape_full5x5_sigmaIphiIphi',
   'eid_shape_full5x5_circularity',
   'eid_shape_full5x5_r9',
   'eid_sc_etaWidth',
   'eid_sc_phiWidth',
   'eid_shape_full5x5_HoverE',
   'eid_trk_nhits',
   'eid_trk_chi2red',
   'eid_gsf_chi2red',
   'eid_brem_frac',
   'eid_gsf_nhits',
   'eid_match_SC_EoverP',
   'eid_match_eclu_EoverP',
   'eid_match_SC_dEta',
   'eid_match_SC_dPhi',
   'eid_match_seed_dEta',
   'eid_sc_E',
   'eid_trk_p',
   'gsf_bdtout1'
]

additional = [
   'gen_pt','gen_eta', 
   'trk_pt','trk_eta','trk_charge','trk_dr',
   'gsf_pt','gsf_eta','gsf_dr','gsf_bdtout2','gsf_mode_pt',
   'ele_pt','ele_eta','ele_dr',
   'ele_mva_value','ele_mva_value_retrained',#'ele_mva_value_old',
   'ele_mva_value_depth10','ele_mva_value_depth11','ele_mva_value_depth13',#'ele_mva_value_depth15',
   'evt','weight','rho',
   'tag_pt','tag_eta',
   'gsf_dxy','gsf_dz','gsf_nhits','gsf_chi2red',
]

pfgsf_branches = [
    'has_pfgsf','pfgsf_pt','pfgsf_eta',
]

labelling = [
    'is_mc',
    'is_e','is_egamma',
    'has_trk','has_seed','has_gsf','has_ele',
    'seed_trk_driven','seed_ecal_driven'
]

columns = features + additional + pfgsf_branches + labelling
columns = list(set(columns))
