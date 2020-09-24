from fnmatch import fnmatch
class RangesByName(object):
   def __init__(self, rlist):
      self._rlist_ = rlist #list of (ending, range)
      self._rlist_.sort(key=lambda x: -1*len(x[0]))

   def get(self, val, default=None):
      for ending, vrange in self._rlist_:
         if fnmatch(val, ending):
            return vrange
      return default

   def __getitem__(self, val):
      return self.get(val)

ranges = RangesByName([
   ('*_cluster_deta', (-2, 2)),
   ('*_pt', (0, 15)),
   ('*_dpt', (0, 2)),
   ('*_eta' , (-3, 3)),
   ('*_inp' , (0, 20)),
   ('*_outp' , (0, 10)),
   ('*_chi2red' , (0, 6)),
   ('*_Deta' , (0, 0.2)),
   ('*_Dphi' , (-0.2, 0.2)),
   ('*_nhits' , (0, 50)),
   ('*_p' , (0, 20)),
   ('*_cluster_e', (0, 20)),
   ('*_cluster_ecorr', (0, 20)),
   ('*_cluster_eta', (-3, 3)),
   ('*_cluster_deta', (-1.5, 1.5)),
   ('*_cluster_dphi', (-1.5, 1.5)),
   ('*brem_frac', (0,1)),
   ('*_fracSC', (0.4, 1)),
   ('*brem_fracTrk', (-4,2)),
   ('*_covEtaEta', (-1, 1)),
   ('*_covEtaPhi', (-1, 1)),
   ('*_cluster_covPhiPhi', (-1, 1)),
   ('*_EoverP', (0, 2)),
   ('*_dEta', (-1, 1)),
   ('*_dPhi', (-0.5, 0.5)),
   ('*EoverPout', (0, 50)),
   ('*dEta_vtx', (-5, 5)),
   ('*sc_E', (0, 20)),
   ('*sc_Et', (0, 200)),
   ('*sc_RawE', (0, 100)),
   ('*sc_etaWidth', (0, 0.05)),
   ('*sc_phiWidth', (0, 0.1)),
   ('*HoverE', (0,1)),
   ('*HoverEBc', (0, 0.4)),
   ('*_e[0-9]x[0-5]*', (0, 10.)),
   ('shape_e[BLRT]*', (0, 2.)),
   ('*_full5x5_HoverE', (0, 2)),
   ('*_full5x5_HoverEBc', (0, 1)),
   ('*full5x5_e[BLRT]*', (0, 2)),
   ('*_hcalDepth1*', (0, 1)),
   ('*_hcalDepth2*', (0, 0.3)),
   ('*_r9', (0., 1.2)),
   ('*_sigmaEtaEta', (0., 0.1)),
   ('*_sigmaIetaIeta', (0., 0.05)),
   ('*_sigmaIphiIphi', (0., 0.06)),
   ('*trk_dxy_sig', (-3,3)),
   ('*_gsf_chiratio', (0, 2)),
   ('trk_dxy_sig_inverted', (-3, 3))
   ## ('*', ()),
])

beauty = {
    'gen_pt' : r'p$_T$(gen)', 
    'gen_eta' : r'$\eta$(gen)', 
    'dxy_err' : r'$\sigma$(dxy)', 
    'nhits' : r'\# of hits',
    'trk_pt' : r'p$_T$(ktf track)', 
    'trk_eta' : r'$\eta$(ktf track)', 
    'trk_ieta' : r'$i\eta$(ktf track)',
    'log_trkpt' : r'log$_{10}$(p$_T$)(ktf track)', 
    'trk_inp' : r'p$_{in}$(ktf track)',
    'trk_outp' : r'p$_{out}$(ktf track)', 
    'trk_eta' : r'$\eta$(ktf track)', 
    'trk_ecal_Deta': '$\Delta\eta$(ECAL, ktf track)',
    'trk_ecal_Dphi' : '$\Delta\varphi$(ECAL, ktf track)',
    'e_over_p' : 'E/p', 
    'trk_chi2red' : '$\chi^2$(ktf track)/ndf', 
    'gsf_dpt' : r'p$_T$(gsf track)',
    'trk_gsf_chiratio' : '$\chi^2$(gsf track)/$\chi^2$(ktf track)', 
    'gsf_chi2red' : '$\chi^2$(gsf track)/ndf', 
    'xy_sig' : r'$\sigma$(dxy)/dxy',
}
