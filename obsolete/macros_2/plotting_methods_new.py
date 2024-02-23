import numpy as np

from plotting import plot

################################################################################
# 
def AxE_retraining(plt,df_lowpt,df_egamma) :

   print 
   print "AxE_retraining"
   
   # Low-pT GSF electrons (Unbiased seed BDT)
   has_gsf = (df_lowpt.has_gsf) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   plot( plt=plt, df=df_lowpt, string="Low pT GSF trk (PreId), AxE",
         selection=has_gsf, draw_roc=True, draw_eff=False,
         label='Low-$p_{T}$ GSF track',
         color='red', markerstyle='^', markersize=12, linestyle='dashed',
         discriminator=df_lowpt.gsf_bdtout1,
   )
   
   # Low-pT GSF electrons (2019Jun28)
   has_gsf = (df_lowpt.has_gsf) & (df_lowpt.gsf_pt>0.5) & (np.abs(df_lowpt.gsf_eta)<2.4)
   has_ele = (df_lowpt.has_ele) & (df_lowpt.ele_pt>0.5) & (np.abs(df_lowpt.ele_eta)<2.4)
   eff2,fr2,roc2 = plot( plt=plt, df=df_lowpt, string="Low pT ele (latest), AxE",
                         selection=has_ele, draw_roc=True, draw_eff=False,
                         label='Low-$p_{T}$ electron',
                         color='blue', markerstyle='o', markersize=12, linestyle='solid',
                         discriminator=df_lowpt.training_out,
   )

   # EGamma PF GSF track 
   has_gsf = (df_egamma.has_gsf) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   has_pfgsf = (df_egamma.has_pfgsf) & (df_egamma.pfgsf_pt>0.5) & (np.abs(df_egamma.pfgsf_eta)<2.4)
   eff1,fr1,_ = plot( plt=plt, df=df_egamma, string="EGamma GSF trk, AxE",
                      selection=has_pfgsf, draw_roc=False, draw_eff=True,
                      label='Standard GSF track',
                      color='orange', markerstyle='^', markersize=12, linestyle='',
   )
   
   # EGamma PF ele 
   has_gsf = (df_egamma.has_gsf) & (df_egamma.gsf_pt>0.5) & (np.abs(df_egamma.gsf_eta)<2.4)
   has_ele = (df_egamma.has_ele) & (df_egamma.ele_pt>0.5) & (np.abs(df_egamma.ele_eta)<2.4)
   plot( plt=plt, df=df_egamma, string="EGamma PF ele, AxE",
         selection=has_ele, draw_roc=False, draw_eff=True,
         label='Standard PF electron',
         color='purple', markerstyle='o', markersize=12, linestyle='',
   )

#   roc = (roc2[0]*fr2,roc2[1]*eff2,roc2[2]) 
#   idxL = np.abs(roc[0]-fr1).argmin()
#   idxT = np.abs(roc[1]-eff1).argmin()
#   print "   PFele: eff/fr/thresh:",\
#      "{:.3f}/{:.4f}/{:4.2f} ".format(eff1,fr1,np.nan)
#   print "   Loose: eff/fr/thresh:",\
#      "{:.3f}/{:.4f}/{:4.2f} ".format(roc[1][idxL],roc[0][idxL],roc[2][idxL])
#   print "   Tight: eff/fr/thresh:",\
#      "{:.3f}/{:.4f}/{:4.2f} ".format(roc[1][idxT],roc[0][idxT],roc[2][idxT])
