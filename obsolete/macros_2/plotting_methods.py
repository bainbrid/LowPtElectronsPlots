import numpy as np

from plotting import plot

################################################################################
# AxE(Low-pT GSF + PreId) vs KF tracks, OR
# AxE(Low-pT GSF + PreId) vs GSF tracks
def AxE(plt,df_lowpt,df_egamma,df_orig=None) :

   xaxis_egamma = None 
   xaxis_lowpt = None 
   if df_orig is not None : 
      xaxis_egamma = df_orig[(df_orig.has_gsf) & \
                                (df_orig.gsf_pt>0.5) & \
                                (np.abs(df_orig.gsf_eta)<2.4)]
      xaxis_lowpt = df_lowpt[(df_lowpt.has_gsf) & \
                                (df_lowpt.gsf_pt>0.5) & \
                                (np.abs(df_lowpt.gsf_eta)<2.4)]
      
   # EGamma GSF 
   print 
   has_gsf = (df_egamma.has_gsf) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   plot( plt=plt, df=df_egamma, string="EGamma GSF tracks, AxE",
         selection=has_gsf, draw_roc=False, draw_eff=True,
         label='EGamma GSF ($\mathcal{A}\epsilon$)',
         color='green', markersize=8, linestyle='solid',
         df_xaxis=xaxis_egamma
         )

   # EGamma PF ele 
   print 
   has_ele = (df_egamma.has_ele) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   plot( plt=plt, df=df_egamma, string="EGamma PF ele, AxE",
         selection=has_ele, draw_roc=False, draw_eff=True,
         label='EGamma PF ele ($\mathcal{A}\epsilon$)',
         color='purple', markersize=8, linestyle='solid',
         df_xaxis=xaxis_egamma
         )

   # Low-pT GSF tracks (unbiased + ptbiased)
   print 
   has_gsf = (df_lowpt.has_gsf) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT GSF tracks + unbiased, AxE",
         selection=has_gsf, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ GSF + PreId ($\mathcal{A}\epsilon$)',
         color='red', markersize=8, linestyle='solid',
         discriminator=df_lowpt.gsf_bdtout1,
         df_xaxis=xaxis_lowpt
         )
   plot( plt=plt, df=df_lowpt, string="Low pT GSF tracks + ptbiased, AxE",
         selection=has_gsf, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ GSF + PreId ($\mathcal{A}\epsilon$)',
         color='red', markersize=8, linestyle='dashed',
         discriminator=df_lowpt.gsf_bdtout2,
         df_xaxis=xaxis_lowpt
         )

   # Low-pT GSF electrons (unbiased + ptbiased)
   print 
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele + unbiased, AxE",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + PreId ($\mathcal{A}\epsilon$)',
         color='blue', markersize=8, linestyle='solid',
         discriminator=df_lowpt.gsf_bdtout1,
         df_xaxis=xaxis_lowpt
         )

   plot( plt=plt, df=df_lowpt, string="Low pT ele + ptbiased, AxE",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + PreId ($\mathcal{A}\epsilon$)',
         color='blue', markersize=8, linestyle='dashed',
         discriminator=df_lowpt.gsf_bdtout2,
         df_xaxis=xaxis_lowpt
         )

################################################################################
# AxE(Low-pT GSF + PreId) vs KF tracks
# AxE(Low-pT ele + CMSSW ID) vs KF tracks
def AxE_cmssw_id(plt,df_lowpt,df_egamma) :
      
   # EGamma GSF 
   print 
   has_gsf = (df_egamma.has_gsf) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   plot( plt=plt, df=df_egamma, string="EGamma GSF tracks, AxE",
         selection=has_gsf, draw_roc=False, draw_eff=True,
         label='EGamma GSF ($\mathcal{A}\epsilon$)',
         color='green', markersize=8, linestyle='solid',
         )

   # EGamma PF ele 
   print 
   has_ele = (df_egamma.has_ele) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   plot( plt=plt, df=df_egamma, string="EGamma PF ele, AxE",
         selection=has_ele, draw_roc=False, draw_eff=True,
         label='EGamma PF ele ($\mathcal{A}\epsilon$)',
         color='purple', markersize=8, linestyle='solid',
         )

   # Low-pT GSF tracks (unbiased + ptbiased)
   print 
   has_gsf = (df_lowpt.has_gsf) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT GSF tracks + unbiased, AxE",
         selection=has_gsf, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ GSF + PreId ($\mathcal{A}\epsilon$)',
         color='red', markersize=8, linestyle='solid',
         discriminator=df_lowpt.gsf_bdtout1,
         )
   plot( plt=plt, df=df_lowpt, string="Low pT GSF tracks + ptbiased, AxE",
         selection=has_gsf, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ GSF + PreId ($\mathcal{A}\epsilon$)',
         color='red', markersize=8, linestyle='dashed',
         discriminator=df_lowpt.gsf_bdtout2,
         )

   # Low-pT GSF electrons (CMSSW ID)
   print 
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele + unbiased, AxE",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + CMSSW ID ($\mathcal{A}\epsilon$)',
         color='blue', markersize=8, linestyle='solid',
         discriminator=df_lowpt.ele_mva_value,
         )

################################################################################
# AxE(Low-pT GSF + PreId) vs KF tracks
# AxE(Low-pT ele + CMSSW ID) vs KF tracks
# *** Requires option: --load_model ***
def AxE_seed(plt,df_lowpt,df_egamma) :
     
   # EGamma PF ele + ID
   print 
   has_ele = (df_egamma.has_ele) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   plot( plt=plt, df=df_egamma, string="EGamma PF ele + ID, AxE",
         selection=has_ele, draw_roc=True, draw_eff=True,
         label='EGamma PF ele + ID ($\mathcal{A}\epsilon$)',
         color='purple', markersize=8, linestyle='solid',
         discriminator=df_egamma.ele_mva_value,
         mask = df_egamma.has_seed
         )
   
   # EGamma PF ele (tracker driven) + ID
   print 
   has_ele = (df_egamma.has_ele) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   tracker_driven = (df_egamma.seed_trk_driven==True)
   plot( plt=plt, df=df_egamma, string="EGamma PF ele (tracker-driven) + ID, AxE",
         selection = has_ele & tracker_driven, 
         draw_roc=True, draw_eff=True,
         label='EGamma PF ele (tracker-driven) + ID ($\mathcal{A}\epsilon$)',
         color='purple', markersize=8, linestyle='dashed',
         discriminator=df_egamma.ele_mva_value,
         mask = df_egamma.has_seed
         )

   # EGamma PF ele + ID
   print 
   has_ele = (df_egamma.has_ele) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   ecal_driven_only = (df_egamma.seed_ecal_driven==True) & (df_egamma.seed_trk_driven==False)
   plot( plt=plt, df=df_egamma, string="EGamma PF ele (ECAL driven) + ID, AxE",
         selection = has_ele & ecal_driven_only,
         draw_roc=True, draw_eff=True,
         label='EGamma PF ele (ECAL-driven) + ID ($\mathcal{A}\epsilon$)',
         color='purple', markersize=8, linestyle='dashdot',
         discriminator=df_egamma.ele_mva_value,
         mask = df_egamma.has_seed
         )

################################################################################
# 
def AxE_retraining(plt,df_lowpt,df_egamma) :

   has_gsf = (df_egamma.has_gsf) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   has_ele = (df_egamma.has_ele) & (df_egamma.ele_pt>0.5) & (np.abs(df_egamma.ele_eta)<2.4)

   # EGamma PF ele 
   print 
   eff1,fr1,_ = plot( plt=plt, df=df_egamma, string="EGamma GSF trk, AxE",
                      selection=has_gsf, draw_roc=False, draw_eff=True,
                      label='EGamma GSF track ($\mathcal{A}\epsilon$)',
                      color='green', markersize=8, linestyle='solid',
                   )

   # EGamma PF ele 
   print 
   plot( plt=plt, df=df_egamma, string="EGamma PF ele, AxE",
         selection=has_ele, draw_roc=False, draw_eff=True,
         label='EGamma PF ele ($\mathcal{A}\epsilon$)',
         color='purple', markersize=8, linestyle='solid',
         )

   has_gsf = (df_lowpt.has_gsf) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.ele_pt>0.5) & (np.abs(df_lowpt.ele_eta)<2.4)

   # Low-pT GSF electrons (PreId unbiased)
   print 
   plot( plt=plt, df=df_lowpt, string="Low pT GSF trk (PreId), AxE",
         selection=has_gsf, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ GSF track + unbiased ($\mathcal{A}\epsilon$)',
         color='red', markersize=8, linestyle='dashed',
         discriminator=df_lowpt.gsf_bdtout1,
         )

   # Low-pT GSF electrons (CMSSW)
   print 
   has_ele = (df_lowpt.has_ele) & (df_lowpt.ele_pt>0.5) & (np.abs(df_lowpt.ele_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (CMSSW), AxE",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + 2019Jun28 model ($\mathcal{A}\epsilon$)',
         color='blue', markersize=8, linestyle='dashdot',
         discriminator=df_lowpt.ele_mva_value,
         )

   # Low-pT GSF electrons (retraining)
   print 
   eff2,fr2,roc2 = plot( plt=plt, df=df_lowpt, string="Low pT ele (latest), AxE",
                         selection=has_ele, draw_roc=True, draw_eff=False,
                         label='Low-$p_{T}$ ele + latest model ($\mathcal{A}\epsilon$)',
                         color='blue', markersize=8, linestyle='solid',
                         discriminator=df_lowpt.training_out,
                      )

   roc = (roc2[0]*fr2,roc2[1]*eff2,roc2[2]) 
   idxL = np.abs(roc[0]-fr1).argmin()
   idxT = np.abs(roc[1]-eff1).argmin()
   print "   PFele: eff/fr/thresh:",\
      "{:.3f}/{:.4f}/{:4.2f} ".format(eff1,fr1,np.nan)
   print "   Loose: eff/fr/thresh:",\
      "{:.3f}/{:.4f}/{:4.2f} ".format(roc[1][idxL],roc[0][idxL],roc[2][idxL])
   print "   Tight: eff/fr/thresh:",\
      "{:.3f}/{:.4f}/{:4.2f} ".format(roc[1][idxT],roc[0][idxT],roc[2][idxT])

################################################################################
# 
def AxE_retraining_binned1(plt,df_lowpt,df_egamma) :

   # EGamma PF ele 
   print 
   has_ele = (df_egamma.has_ele) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   plot( plt=plt, df=df_egamma, string="EGamma PF ele, AxE",
         selection=has_ele, draw_roc=False, draw_eff=True,
         label='EGamma PF ele ($\mathcal{A}\epsilon$)',
         color='purple', markersize=8, linestyle='solid',
         )

   # Low-pT GSF electrons (retraining, pT>2.0)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (retraining, pT>2.0), AxE",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + retrained model, $p^{trk}_{T} > 2.0\, GeV$',
         color='blue', markersize=8, linestyle='solid',
         discriminator=df_lowpt.training_out,
         mask = has_trk & (df_lowpt.trk_pt>2.0),
         )

   # Low-pT GSF electrons (retraining, 0.5<pT<2.0)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (retraining, 0.5<pT<2.0)",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + retrained model, $0.5 < p^{trk}_{T} < 2.0\, GeV$',
         color='blue', markersize=8, linestyle='dashed',
         discriminator=df_lowpt.training_out,
         mask = has_trk & (df_lowpt.trk_pt>0.5) & (df_lowpt.trk_pt<2.0),
         )

   # Low-pT GSF electrons (CMSSW, pT>2.0)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (CMSSW, pT>2.0)",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + CMSSW model, $p^{trk}_{T} > 2.0\, GeV$',
         color='red', markersize=8, linestyle='solid',
         discriminator=df_lowpt.ele_mva_value,
         mask = has_trk & (df_lowpt.trk_pt>2.0),
         )

   # Low-pT GSF electrons (CMSSW, 0.5<pT<2.0)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (CMSSW, 0.5<pT<2.0)",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + CMSSW model, $0.5 < p^{trk}_{T} < 2.0\, GeV$',
         color='red', markersize=8, linestyle='dashed',
         discriminator=df_lowpt.ele_mva_value,
         mask = has_trk & (df_lowpt.trk_pt>0.5) & (df_lowpt.trk_pt<2.0),
         )

################################################################################
# 
def AxE_retraining_binned2(plt,df_lowpt,df_egamma) :

   # EGamma PF ele 
   print 
   has_ele = (df_egamma.has_ele) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   plot( plt=plt, df=df_egamma, string="EGamma PF ele, AxE",
         selection=has_ele, draw_roc=False, draw_eff=True,
         label='EGamma PF ele ($\mathcal{A}\epsilon$)',
         color='purple', markersize=8, linestyle='solid',
         )

   ##########

   # Low-pT GSF electrons (seeding, pT>2.0)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (seeding, pT>2.0), AxE",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + seed BDT, $p^{trk}_{T} > 2.0\, GeV$',
         color='green', markersize=8, linestyle='solid',
         discriminator=df_lowpt.gsf_bdtout1,
         mask = (df_lowpt.gsf_pt>2.0),
         )

   # Low-pT GSF electrons (seeding, 1.5<pT<2.0)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (seeding, 1.5<pT<2.0)",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + seed BDT, $0.5 < p^{trk}_{T} < 2.0\, GeV$',
         color='green', markersize=8, linestyle='dashed',
         discriminator=df_lowpt.gsf_bdtout1,
         mask = (df_lowpt.gsf_pt>1.5) & (df_lowpt.gsf_pt<2.0),
         )

   # Low-pT GSF electrons (seeding, 1.5<pT<1.5)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (seeding, 1.0<pT<1.5)",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + seed BDT, $0.5 < p^{trk}_{T} < 2.0\, GeV$',
         color='green', markersize=8, linestyle='dashdot',
         discriminator=df_lowpt.gsf_bdtout1,
         mask = (df_lowpt.gsf_pt>1.0) & (df_lowpt.gsf_pt<1.5),
         )

   # Low-pT GSF electrons (seeding, 0.5<pT<1.0)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (seeding, 0.5<pT<1.0)",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + seed BDT, $0.5 < p^{trk}_{T} < 2.0\, GeV$',
         color='green', markersize=8, linestyle='dotted',
         discriminator=df_lowpt.gsf_bdtout1,
         mask = (df_lowpt.gsf_pt>0.5) & (df_lowpt.gsf_pt<1.0),
         )

   ##########

   # Low-pT GSF electrons (retraining, pT>2.0)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (retraining, pT>2.0), AxE",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + retrained model, $p^{trk}_{T} > 2.0\, GeV$',
         color='blue', markersize=8, linestyle='solid',
         discriminator=df_lowpt.training_out,
         mask = (df_lowpt.gsf_pt>2.0),
         )

   # Low-pT GSF electrons (retraining, 1.5<pT<2.0)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (retraining, 1.5<pT<2.0)",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + retrained model, $0.5 < p^{trk}_{T} < 2.0\, GeV$',
         color='blue', markersize=8, linestyle='dashed',
         discriminator=df_lowpt.training_out,
         mask = (df_lowpt.gsf_pt>1.5) & (df_lowpt.gsf_pt<2.0),
         )

   # Low-pT GSF electrons (retraining, 1.5<pT<1.5)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (retraining, 1.0<pT<1.5)",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + retrained model, $0.5 < p^{trk}_{T} < 2.0\, GeV$',
         color='blue', markersize=8, linestyle='dashdot',
         discriminator=df_lowpt.training_out,
         mask = (df_lowpt.gsf_pt>1.0) & (df_lowpt.gsf_pt<1.5),
         )

   # Low-pT GSF electrons (retraining, 0.5<pT<1.0)
   print 
   has_trk = (df_lowpt.has_trk) & (df_lowpt.trk_pt>0.5) & (np.abs(df_lowpt.trk_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT ele (retraining, 0.5<pT<1.0)",
         selection=has_ele, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ ele + retrained model, $0.5 < p^{trk}_{T} < 2.0\, GeV$',
         color='blue', markersize=8, linestyle='dotted',
         discriminator=df_lowpt.training_out,
         mask = (df_lowpt.gsf_pt>0.5) & (df_lowpt.gsf_pt<1.0),
         )
