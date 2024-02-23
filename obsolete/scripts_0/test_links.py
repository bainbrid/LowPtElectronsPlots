import ROOT
import pprint
from DataFormats.FWLite import Events, Handle
from pdb import set_trace
import numpy as np
import pandas as pd
import itertools
ROOT.gROOT.SetBatch()

#events = Events('root://cmsxrootd.fnal.gov//store/data/Run2018A/ParkingBPH1/MINIAOD/22Mar2019-v1/260002/54516928-947E-6140-A489-4E4099A593CF.root')
#events, aod = Events('/afs/cern.ch/user/b/bainbrid/www/ForMauro/step3_inMINIAODSIM_fromAODSIM.root'), False
#events, aod = Events('/afs/cern.ch/user/b/bainbrid/www/ForMauro/step3_inAODSIM.root'), True
events, aod = Events('/afs/cern.ch/user/b/bainbrid/www/ForMauro/step3_inMINIAODSIM_fromAODSIM_3.root'), False
#iterator = events.__iter__()
hass1 = Handle('edm::Association<vector<pat::PackedCandidate> >')
hass2 = Handle('edm::Association<vector<pat::PackedCandidate> >')
hass3 = Handle('edm::Association<vector<reco::Track> >')
hgsf = Handle('vector<reco::GsfTrack>')

df = pd.DataFrame({
    'g_pt'  : [],
    'g_eta' : [],
    'g_phi' : [],
    'm_pt'  : [],
    'm_eta' : [],
    'm_phi' : [],
})
#evt = iterator.next()
for evt, II in itertools.izip(events, range(10)):
    print II
    evt.getByLabel('lowPtGsfEleGsfTracks', hgsf)
    if aod:
        evt.getByLabel('lowPtGsfToTrackLinks', hass3)
        associated = [(v, hass3.product().get(idx)) for idx, v in enumerate(hgsf.product())]
    else:
        evt.getByLabel('lowPtGsfLinks:lostTracks', hass1)
        evt.getByLabel('lowPtGsfLinks:packedCandidates', hass2)
        associated = [(v, hass1.product().get(idx), hass2.product().get(idx)) for idx, v in enumerate(hgsf.product())]
        associated = [(i, j if j.isNonnull() else k) for i,j,k in associated]

    df = pd.concat((
        df,
        pd.DataFrame({
            'g_pt'  : np.array([i.pt()  for i, _ in associated]),
            'g_eta' : np.array([i.phi() for i, _ in associated]),
            'g_phi' : np.array([i.eta() for i, _ in associated]),
            'm_pt'  : np.array([i.get().pt()  for _, i in associated]),
            'm_eta' : np.array([i.get().phi() for _, i in associated]),
            'm_phi' : np.array([i.get().eta() for _, i in associated]),
        })
    ))

import math
df['deta'] = np.abs(df.g_eta - df.m_eta)
df['dphis'] = (df.g_phi - df.m_phi + math.pi) % (2*math.pi) - math.pi
df['dr'] = np.sqrt(df['deta']**2 + df['dphis']**2)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.hist(df['dr'])
#plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.ylim(0.1, 10**3)
plt.savefig('links.png')

wdeta = np.abs(df.g_eta + df.m_eta).values
plt.clf()
plt.hist(wdeta[df.dr > 2])
plt.gca().set_yscale('log')
plt.ylim(0.1, 10**4)
plt.savefig('etasum.png')
