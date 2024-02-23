import pandas as pd
import numpy as np

cuts_df = pd.read_csv(
   '../../../RecoParticleFlow/PFTracking/data/Threshold.dat', 
   delim_whitespace=True, header=None
)

def baseline(df):
   matching = df.preid_trk_ecal_match
   bdt_cuts = []
   for ibin in range(9):
      mask = ((df.preid_ibin/9) == ibin)
      c_deta   = cuts_df.loc[ibin][0]
      c_dphi   = cuts_df.loc[ibin][1]
      c_ep     = cuts_df.loc[ibin][2]
      c_hits   = cuts_df.loc[ibin][3]
      c_chimin = cuts_df.loc[ibin][4]
      c_bdt    = cuts_df.loc[ibin][5]
      togsf = (df.trk_nhits < c_hits) | \
         (df.trk_chi2red > c_chimin)
      bdt_sel = (np.invert(matching) & togsf) & (df.preid_bdtout > c_bdt)
      bdt_cuts.append((bdt_sel & mask))
   bdt_selection = np.logical_or.reduce(bdt_cuts)
   df['baseline'] = (bdt_selection | matching)
   df['cutmatching'] = (matching)
   df['cutbdt'] = (bdt_sel)

