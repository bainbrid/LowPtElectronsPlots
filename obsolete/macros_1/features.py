trk_features = [
   'trk_pt',
   'trk_eta',
   'trk_phi',
   'trk_p',
   # 'trk_charge',
   'trk_nhits',
   'trk_high_purity',
   # 'trk_inp',
   # 'trk_outp',
   'trk_chi2red',
   ]

seed_features = trk_features + [
   'preid_trk_ecal_Deta',
   'preid_trk_ecal_Dphi',
   'preid_e_over_p',
]

improved_seed_features = trk_features + [
   'rho',
   'ktf_ecal_cluster_e',
   'ktf_ecal_cluster_deta',
   'ktf_ecal_cluster_dphi',
   'ktf_ecal_cluster_e3x3',
   'ktf_ecal_cluster_e5x5',
   'ktf_ecal_cluster_covEtaEta',
   'ktf_ecal_cluster_covEtaPhi',
   'ktf_ecal_cluster_covPhiPhi',
   'ktf_ecal_cluster_r9',
   'ktf_ecal_cluster_circularity_',
   'ktf_hcal_cluster_e',
   'ktf_hcal_cluster_deta',
   'ktf_hcal_cluster_dphi',
]

seed_gsf = [
   'preid_gsf_dpt',
   'preid_trk_gsf_chiratio',
   'preid_gsf_chi2red',]

fullseed_features = seed_features + seed_gsf
improved_fullseed_features = improved_seed_features + seed_gsf
#'preid_numGSF', should be used as weight?

cmssw_improvedfullseeding = [
  'preid_trk_pt',
  'preid_trk_eta',
  'preid_trk_phi',
  'preid_trk_p',
  'preid_trk_nhits',
  'preid_trk_high_quality',
  'preid_trk_chi2red',
  'preid_rho',
  'preid_ktf_ecal_cluster_e',
  'preid_ktf_ecal_cluster_deta',
  'preid_ktf_ecal_cluster_dphi',
  'preid_ktf_ecal_cluster_e3x3',
  'preid_ktf_ecal_cluster_e5x5',
  'preid_ktf_ecal_cluster_covEtaEta',
  'preid_ktf_ecal_cluster_covEtaPhi',
  'preid_ktf_ecal_cluster_covPhiPhi',
  'preid_ktf_ecal_cluster_r9',
  'preid_ktf_ecal_cluster_circularity',
  'preid_ktf_hcal_cluster_e',
  'preid_ktf_hcal_cluster_deta',
  'preid_ktf_hcal_cluster_dphi',
  'preid_gsf_dpt',
  'preid_trk_gsf_chiratio',
  'preid_gsf_chi2red',
]
#  preid_trk_dxy_sig

cmssw_displaced_improvedfullseeding = cmssw_improvedfullseeding + ['preid_trk_dxy_sig']
cmssw_displaced_improvedfullseeding_fixSIP = cmssw_improvedfullseeding + ['trk_dxy_sig']
cmssw_displaced_improvedfullseeding_fixInvSIP = cmssw_improvedfullseeding + ['trk_dxy_sig_inverted']

id_features = trk_features + [
   'gsf_pt',
   'gsf_eta',
   'gsf_phi',
   'gsf_p',
   'gsf_charge',
   'gsf_nhits',
   'gsf_inp',
   'gsf_outp',
   'gsf_chi2red',

   'gsf_ecal_cluster_e',
   'gsf_ecal_cluster_ecorr',
   'gsf_ecal_cluster_eta',
   'gsf_ecal_cluster_deta',
   'gsf_ecal_cluster_dphi',
   'gsf_ecal_cluster_covEtaEta',
   'gsf_ecal_cluster_covEtaPhi',
   'gsf_ecal_cluster_covPhiPhi',
   'gsf_hcal_cluster_e',
   'gsf_hcal_cluster_eta',
   'gsf_hcal_cluster_deta',
   'gsf_hcal_cluster_dphi',

   'gsf_ktf_same_ecal',
   'gsf_ktf_same_hcal',

   'ktf_ecal_cluster_e',
   'ktf_ecal_cluster_ecorr',
   'ktf_ecal_cluster_eta',
   'ktf_ecal_cluster_deta',
   'ktf_ecal_cluster_dphi',
   'ktf_ecal_cluster_covEtaEta',
   'ktf_ecal_cluster_covEtaPhi',
   'ktf_ecal_cluster_covPhiPhi',
   'ktf_hcal_cluster_e',
   'ktf_hcal_cluster_eta',
   'ktf_hcal_cluster_deta',
   'ktf_hcal_cluster_dphi',
]

new_features = [
   'match_SC_EoverP',
   'match_SC_dEta',
   'match_SC_dPhi',
   'match_seed_EoverP',
   'match_seed_EoverPout',
   'match_seed_dEta',
   'match_seed_dPhi',
   'match_seed_dEta_vtx',
   'match_eclu_EoverP',
   'match_eclu_dEta',
   'match_eclu_dPhi',
   
   'shape_sigmaEtaEta',
   'shape_sigmaIetaIeta',
   'shape_sigmaIphiIphi',
   'shape_e1x5',
   'shape_e2x5Max',
   'shape_e5x5',
   'shape_r9',
   'shape_HoverE',
   'shape_HoverEBc',
   'shape_hcalDepth1',
   'shape_hcalDepth2',
   'shape_hcalDepth1Bc',
   'shape_hcalDepth2Bc',
   'shape_nHcalTowersBc',
   'shape_eLeft',
   'shape_eRight',
   'shape_eTop',
   'shape_eBottom',
   'shape_full5x5_sigmaEtaEta',
   'shape_full5x5_sigmaIetaIeta',
   'shape_full5x5_sigmaIphiIphi',
   'shape_full5x5_circularity',
   'shape_full5x5_e1x5',
   'shape_full5x5_e2x5Max',
   'shape_full5x5_e5x5',
   'shape_full5x5_r9',
   'shape_full5x5_HoverE',
   'shape_full5x5_HoverEBc',
   'shape_full5x5_hcalDepth1',
   'shape_full5x5_hcalDepth2',
   'shape_full5x5_hcalDepth1Bc',
   'shape_full5x5_hcalDepth2Bc',
   'shape_full5x5_eLeft',
   'shape_full5x5_eRight',
   'shape_full5x5_eTop',
   'shape_full5x5_eBottom',
   
   'brem_frac',
   'brem_fracTrk',
   'brem_fracSC',
   'brem_N',
   
   'sc_etaWidth',
   'sc_phiWidth',
   'sc_ps_EoverEraw',
   'sc_E',
   'sc_Et',
   'sc_eta',
   'sc_phi',
   'sc_RawE',
   'sc_Nclus',
]

seed_additional = ['trk_pass_default_preid', 'preid_bdtout1', 'preid_bdtout2']
seed_94X_additional = ['preid_trk_ecal_match', 'preid_trkfilter_pass', 'preid_mva_pass']
#id_additional = ['ele_mvaIdV2', 'ele_lowPtMva', 'ele_pt']
id_additional = ['ele_mva_value', 'ele_mva_id', 'ele_pt'] #@@

labeling = ['is_e', 'is_e_not_matched', 'is_other', # original labels
            'is_egamma', 'has_trk', 'has_seed', 'has_gsf', 'has_ele', ] #@@ new

gen_features = [
   'gen_pt',
   'gen_eta',
   'gen_phi',
   'gen_charge',
   ]

mva_id_inputs = [
   'rho',
   'ele_pt',
   'sc_eta',
   'shape_full5x5_sigmaIetaIeta',
   'shape_full5x5_sigmaIphiIphi',
   'shape_full5x5_circularity',
   'shape_full5x5_r9',
   'sc_etaWidth',
   'sc_phiWidth',
   'shape_full5x5_HoverE',
   'trk_nhits',
   'trk_chi2red',
   'gsf_chi2red',
   'brem_frac',
   'gsf_nhits',
   'match_SC_EoverP',
   'match_eclu_EoverP',
   'match_SC_dEta', #should be abs
   'match_SC_dPhi', #should be abs
   'match_seed_dEta',  #should be abs
   'sc_E',
   'trk_p',
#ele_expected_inner_hits            gsfTrack.hitPattern.numberOfLostHits('MISSING_INNER_HITS') None None
#ele_conversionVertexFitProbability electronMVAVariableHelper:convVtxFitProb                   None None
#ele_IoEmIop                        1.0/ecalEnergy-1.0/trackMomentumAtVtx.R                    None None
]

cmssw_mva_id = [
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
]


to_drop = {
   'gsf_phi',
   'gsf_charge',
   'trk_outp',
   'trk_p',
   'gsf_ecal_cluster_e',
   'gsf_inp',
   'trk_eta'
}

useless = {
   'trk_p',
   'trk_high_purity',
   'gsf_inp',
   'gsf_charge',
   'gsf_ktf_same_hcal',
   'trk_charge',
   'gsf_ktf_same_ecal',
   'ktf_ecal_cluster_covEtaPhi',
   'gsf_ecal_cluster_covEtaEta',
   'gsf_ecal_cluster_covEtaPhi',
   'gsf_ecal_cluster_e',
   'gsf_ecal_cluster_covPhiPhi',
   'shape_full5x5_sigmaIetaIeta',
}

def get_features(ftype):
   add_ons = []
   if ftype.startswith('displaced_'):
      add_ons = ['trk_dxy_sig']
      ftype = ftype.replace('displaced_', '')
   if ftype == 'seeding':
      features = seed_features
      additional = seed_additional
   elif ftype == 'trkonly':
      features = trk_features
      additional = seed_additional
   elif ftype == 'betterseeding':
      features = seed_features+['rho',]
      additional = seed_additional
   elif ftype == 'fullseeding':
      features = fullseed_features
      additional = seed_additional
   elif ftype == 'improvedseeding':
      features = improved_seed_features
      additional = seed_additional
   elif ftype == 'improvedfullseeding':
      features = improved_fullseed_features
      additional = seed_additional
   elif ftype == 'id':
      features = id_features
      additional = id_additional
   elif ftype == 'mva_id':
      features = mva_id_inputs
      additional = id_additional
   elif ftype == 'combined_id':
      features = list(set(mva_id_inputs+id_features))#-to_drop-useless)
      additional = id_additional
   elif ftype == 'cmssw_displaced_improvedfullseeding':
      features = cmssw_displaced_improvedfullseeding
      additional = seed_additional
   elif ftype == 'cmssw_improvedfullseeding':
      features = cmssw_improvedfullseeding
      additional = seed_additional
   elif ftype == 'cmssw_mva_id':
      features = cmssw_mva_id
      additional = id_additional + ['gsf_bdtout1','gsf_bdtout2']
   elif ftype == 'cmssw_mva_id_extended':
      features = cmssw_mva_id + ['gsf_bdtout1']
      additional = id_additional + ['gsf_bdtout2']
   elif ftype == 'cmssw_displaced_improvedfullseeding_fixSIP':
      features = cmssw_displaced_improvedfullseeding_fixSIP
      additional = seed_additional
   elif ftype == 'cmssw_displaced_improvedfullseeding_fixInvSIP':
      features = cmssw_displaced_improvedfullseeding_fixInvSIP
      additional = seed_additional
   elif ftype == 'basic_plots_default':
      features = cmssw_mva_id \
          + cmssw_displaced_improvedfullseeding \
          + ['trk_dxy_sig', 'trk_dxy_sig_inverted'] \
          + ['sc_Nclus', 'ele_eta', 'gsf_eta']
      additional = seed_additional
   else:
      raise ValueError('%s is not among the possible feature collection' % ftype)
   return features+add_ons, additional
