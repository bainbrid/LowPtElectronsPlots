import uproot as uproot
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score

################################################################################
# Variables

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
   #'eid_brem_frac',
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
   'trk_pt','trk_eta','trk_dr','trk_charge',
   'gsf_pt','gsf_eta','gsf_dr','gsf_bdtout2','gsf_mode_pt',
   'ele_pt','ele_eta','ele_dr',
   'ele_mva_value','ele_mva_value_retrained','ele_mva_value_depth10','ele_mva_value_depth15',
   #'evt',
   'weight','rho',
   'tag_pt','tag_eta',
   'gsf_dxy','gsf_dz','gsf_nhits','gsf_chi2red',
]

labelling = [
   'is_e','is_egamma',
   'has_trk','has_seed','has_gsf','has_ele',
]

columns = features + additional + labelling
columns = list(set(columns))

################################################################################
# Parse files

def parse(files,nevents=-1,verbose=False) :
   df = None
   for ifile,file in enumerate(files) : 
      tmp = uproot.open(file).get('ntuplizer/tree').pandas.df(branches=columns)
      #tmp = tmp[np.invert(tmp.is_egamma)&(tmp.has_ele)]
      if verbose : print(f'ifile={ifile:.0f}, file={file:s}, entries={tmp.shape[0]:.0f}')
      df = tmp if ifile == 0 else pd.concat([df,tmp])
      if nevents > 0 and df.shape[0] > nevents :
         df = df.head(nevents)
         print(f"Consider only first {nevents:.0f} events ...")
         break
   return df

################################################################################
# Preprocessing

def preprocess(df) :

   # Filter based on tag muon pT and eta
   tag_muon_pt = 7.0
   tag_muon_eta = 1.5
   df = df[(df.tag_pt>tag_muon_pt)&(np.abs(df.tag_eta)<tag_muon_eta)]
   print(f"tag_muon_pt: {tag_muon_pt:.2f}, "\
            f"tag_muon_eta: {tag_muon_eta:.2f}, "\
            f"df.shape: {df.shape}")

   # Add log(trk_pt)
   log_trk_pt = np.log10(df['trk_pt'])
   log_trk_pt[np.isnan(log_trk_pt)] = -999.
   df['log_trk_pt'] = log_trk_pt

   return df

################################################################################
# ROC curves

def roc_curves(dict={'trn':{'label':None,'score':None,'weight':None},
                     'tst':{'label':None,'score':None,'weight':None},
                     'val':{'label':None,'score':None,'weight':None}},
               title='') :

   # ROC curves
   plt.clf()
   plt.figure(figsize=[8,8])

   x = np.logspace(-4.,0.,100)
   plt.plot(x,x,linestyle='--',color='black',linewidth=1) # by chance

   # Train ROC 
   if dict['trn']['label'] is not None and dict['trn']['score'] is not None :
      fpr,tpr,thresholds = roc_curve(y_true=dict['trn']['label'],
                                     y_score=dict['trn']['score'],
                                     sample_weight=dict['trn']['weight'])
      auc = roc_auc_score(dict['trn']['label'],
                          dict['trn']['score'],
                          sample_weight=dict['trn']['weight'])
      plt.plot(fpr,tpr,label=f'Train (AUC={auc:5.3f})')

   # Test ROC 
   if dict['tst']['label'] is not None and dict['tst']['score'] is not None :
      fpr,tpr,thresholds = roc_curve(y_true=dict['tst']['label'],
                                     y_score=dict['tst']['score'],
                                     sample_weight=dict['tst']['weight'])
      auc = roc_auc_score(dict['tst']['label'],
                          dict['tst']['score'],
                          sample_weight=dict['tst']['weight'])
      plt.plot(fpr,tpr,label=f'Test (AUC={auc:5.3f})')

   # Validation ROC 
   if dict['val']['label'] is not None and dict['val']['score'] is not None :
      fpr,tpr,thresholds = roc_curve(y_true=dict['val']['label'],
                                     y_score=dict['val']['score'],
                                     sample_weight=dict['val']['weight'])
      auc = roc_auc_score(dict['val']['label'],
                          dict['val']['score'],
                          sample_weight=dict['val']['weight'])
      plt.plot(fpr,tpr,label=f'Validation (AUC={auc:5.3f})')

   plt.title(title)
   plt.ylabel('Efficiency')
   plt.xlabel('Mistag rate')
   plt.legend(loc='best')
   plt.xlim(0.,1.)
   plt.ylim(0.,1.)
   plt.savefig(f'plots/{title:s}.png',bbox_inches='tight')
   plt.gca().set_xscale('log')
   plt.xlim(1.e-4,1.)
   plt.savefig(f'plots/{title:s}_logy.png',bbox_inches='tight')
   plt.clf()

################################################################################
# Plots related to weights

def calc_weights(df,nbins,reweight=False,verbose=False,filename='weights') :

   ##### COMMON #####

   import os
   import json
   import joblib
   from sklearn.cluster import MiniBatchKMeans
   from matplotlib import pyplot as plt
   from matplotlib.colors import LogNorm
   from sklearn.model_selection import train_test_split
   import xgboost as xgb
   from utils import roc_curves

   # Weights determined for following kinematical features 
   reweight_features = ['log_trk_pt','trk_eta']

   # Vectorized method
   apply_weight = np.vectorize(lambda x,y : y.get(x),excluded={1})

   ##### DETERMINE CLUSTERING (I.E. BINNING) #####

   success == True
   if recluster == False :
      print(f'Retrieving clusterizer model from "{filename}.pkl"...')
      try : 
         clusterizer = joblib.load(f'{filename}.pkl')
         df['cluster'] = clusterizer.predict(df[reweight_features])
      except : 
         success = False
         print(f'Failed to parse "{filename}.pkl". Reclustering...')
   elif recluster == True or success == False :
      print(f'Train/evaluate clusterizer and store in "{filename}.pkl"')
      clusterizer = MiniBatchKMeans(n_clusters=nbins,verbose=True,)#batch_size=3000,n_jobs=3
      clusterizer.fit(df[reweight_features])
      df['cluster'] = clusterizer.predict(df[reweight_features])
      joblib.dump(clusterizer,f'{filename}.pkl',compress=True)

   ##### ATTEMPT TO READ WEIGHTS FROM FILE #####

   success = True
   if reweight == True :
      if not os.path.isfile(f'{filename}.json') or not os.path.isfile(f'{filename}.pkl') :
         print(f'Unable to find "{filename}.pkl" and "{filename}.json". Recalcing the weights...')
         success = False
      else :
         print(f'Retrieving weights and binning from "{filename}.json" and "{filename}.pkl"...')
         model = joblib.load(f'{filename}.pkl')
         bin = model.predict(df[reweight_features])
         weights_json = json.load(open(f'{filename}.json'))
         weights_lut = {}
         for i in weights_json :
            try : weights_lut[int(i)] = weights_json[i]
            except : success = False
         weights = apply_weight(bin,weights_lut)
         df['weight'] = weights * np.invert(df.is_e) + df.is_e
   if success : return df # else continue onwards

   ##### LEARN BINNING IN KINEMATICAL FEATURES FOR REWEIGHTING #####

   print(f'Determining weights and binning to store in "{filename}.json" and "{filename}.pkl"...')

   # Train
   clusterizer = MiniBatchKMeans(n_clusters=nbins,
                                 #batch_size=3000,
                                 verbose=True,
                                 #n_jobs=3,
                                 ) 
   clusterizer.fit(df[reweight_features])

   # Evaluate
   df['cluster'] = clusterizer.predict(df[reweight_features])

   # Determine weights 
   weights = {}
   counts = {}
   for bin,group in df.groupby('cluster') :
      n_sig = group.is_e.sum()
      n_bkg = np.invert(group.is_e).sum()
      if n_sig == 0 : RuntimeError(f'Bin {bin} has no signal events, reduce the number of bins!')
      if n_bkg == 0 : RuntimeError(f'Bin {bin} has no bkgd events, reduce the number of bins!')
      weights[bin] = float(n_sig)/float(n_bkg) if n_bkg > 0 else 1.
      counts[bin] = min(n_sig,n_bkg)

   # Store weights
   joblib.dump(clusterizer,f'{filename}.pkl',compress=True)
   weights['features'] = reweight_features
   with open(f'{filename}.json','w') as w : json.dump(weights,w)

   # Apply weights
   df['weight'] = np.invert(df.is_e) * apply_weight(df['cluster'],weights) + df.is_e

   # Used to plot the decision boundary (assign colour to each)
   mesh_size = 0.01
   x_min,x_max = df['log_trk_pt'].min()-0.3, df['log_trk_pt'].max()+0.3
   y_min,y_max = df['trk_eta'].min()-0.3, df['trk_eta'].max()+0.3
   xx,yy = np.meshgrid(np.arange(x_min,x_max,mesh_size,dtype=np.float32),
                       np.arange(y_min,y_max,mesh_size,dtype=np.float32))

   # Evaluate (for each point in the mesh)
   bin = clusterizer.predict(np.c_[xx.ravel(),yy.ravel()])

   # Bin boundaries
   Z = bin.reshape(xx.shape)
   plt.figure(figsize=[8,8])
   plt.imshow(
      Z, 
      interpolation='nearest',
      extent=(xx.min(),xx.max(),yy.min(),yy.max()),
      cmap=plt.cm.tab10,#Paired,
      aspect='auto', 
      origin='lower')
   plt.title('Binning')
   plt.xlim(x_min,x_max)
   plt.ylim(y_min,y_max)
   plt.xlabel('log_trk_pt')
   plt.ylabel('trk_eta')
   plt.savefig('plots/weights_binning.png',bbox_inches='tight')
   plt.clf()

   # Weights per bin
   Z = apply_weight(bin,weights).reshape(xx.shape)
   plt.figure(figsize=[8,8])
   plt.imshow(
      Z, 
      interpolation='nearest',
      extent=(xx.min(),xx.max(),yy.min(),yy.max()),
      cmap=plt.cm.coolwarm,#seismic,
      norm=LogNorm(vmin=10**-4, vmax=10**4),
      aspect='auto', 
      origin='lower')
   plt.title('Weights')
   plt.xlim(x_min,x_max)
   plt.ylim(y_min,y_max)
   plt.xlabel('log_trk_pt')
   plt.ylabel('trk_eta')
   plt.colorbar()
   plt.savefig('plots/weights.png',bbox_inches='tight')
   plt.clf()

   # Counts per bin
   Z = apply_weight(bin,counts).reshape(xx.shape)
   plt.figure(figsize=[8,8])
   plt.imshow(
      Z, 
      interpolation='nearest',
      extent=(xx.min(),xx.max(),yy.min(),yy.max()),
      cmap=plt.cm.Reds,#seismic,
      norm=LogNorm(vmin=0.1,vmax=max(counts.values())*10.),
      aspect='auto', 
      origin='lower')
   plt.title('Counts per bin')
   plt.xlim(x_min,x_max)
   plt.ylim(y_min,y_max)
   plt.xlabel('log_trk_pt')
   plt.ylabel('trk_eta')
   plt.colorbar()
   plt.savefig('plots/weights_counts.png',bbox_inches='tight')
   plt.clf()

   # 1D distribution of weights
   plt.figure(figsize=[8,8])
   xmin = df['weight'][df['weight']>0.].min()*0.3
   xmax = df['weight'].max()*3.
   bins = np.logspace(np.log(xmin),np.log(xmax),100)
   entries,_,_ = plt.hist(df['weight'],bins,histtype='stepfilled')
   plt.title('')
   plt.xlabel('Weight')
   plt.ylabel('a.u.')
   plt.xlim(xmin,xmax)
   plt.ylim(0.3,entries.max()*3.)
   plt.gca().set_xscale('log')
   plt.gca().set_yscale('log')
   plt.savefig('plots/weights_distribution.png',bbox_inches='tight')
   plt.clf()

   # Weighted and unweighted distributions of kinematical variables
   for var in reweight_features : #+['trk_pt'] :
      xmin = min(df[df.is_e][var].min(),df[np.invert(df.is_e)][var].min())
      xmax = max(df[df.is_e][var].max(),df[np.invert(df.is_e)][var].max())
      kwargs={'bins':50,'density':True,'histtype':'step','range':(xmin,xmax)}
      plt.hist(df[df.is_e][var],
               color='blue',ls='solid',label='signal',
               **kwargs)
      plt.hist(df[np.invert(df.is_e)][var],
               color='orange',ls='dashed',label='bkgd, unweighted',
               **kwargs)
      plt.hist(df[np.invert(df.is_e)][var],weights=df['weight'][np.invert(df.is_e)],
               color='orange',ls='solid',label='bkgd, weighted',
               **kwargs)
      plt.legend(loc='best')
      plt.xlabel(var)
      plt.ylabel('a.u.')
      plt.gca().set_yscale('log')
      plt.savefig(f'plots/weights_{var:s}.png',bbox_inches='tight')
      plt.clf()

   ##### ROC CURVES FOR KINEMATICAL FEATURES BEFORE REWEIGHTING #####

   # Split data set
   trn,val = train_test_split(df,
                              test_size=0.2,
                              shuffle=True,
                              random_state=0)
   trn,tst = train_test_split(trn,
                              test_size=0.2,
                              shuffle=False,
                              random_state=0)
   if verbose :
      print(f'Data set sizes: trn={trn.shape[0]}, tst={tst.shape[0]}, val={val.shape[0]}')

   # Consider just kinematic variables (and label)
   X_trn,y_trn = trn[reweight_features],trn.is_e.astype(int)
   X_tst,y_tst = tst[reweight_features],tst.is_e.astype(int)
   X_val,y_val = val[reweight_features],val.is_e.astype(int)

   # Train to discriminate based on (kinematical) 'reweight features'
   clf = xgb.XGBClassifier()
   early_stop_kwargs = {'eval_set':[(X_trn,y_trn),(X_tst,y_tst)],
                        'eval_metric':['logloss','auc'],
                        'early_stopping_rounds':10
                        }
   clf.fit(X_trn,y_trn,**early_stop_kwargs)

   # Evaluate
   score_trn = clf.predict_proba(X_trn)[:,-1]
   score_tst = clf.predict_proba(X_tst)[:,-1]
   score_val = clf.predict_proba(X_val)[:,-1]
   
   # ROC curves
   roc_curves(dict={'trn':{'label':y_trn,'score':score_trn,'weight':None},
                    'tst':{'label':y_tst,'score':score_tst,'weight':None},
                    'val':{'label':y_val,'score':score_val,'weight':None}},
              title='weights_roc_pre')
   
   # Retrain to discriminate based on (kinematical) 'reweight features' after reweighting
   clf = xgb.XGBClassifier()
   early_stop_kwargs = {'eval_set':[(X_trn,y_trn),(X_tst,y_tst)],
                        'eval_metric':['logloss','auc'],
                        'early_stopping_rounds':10
                        }
   clf.fit(X_trn,y_trn,**early_stop_kwargs)

   # Evaluate
   score_trn = clf.predict_proba(X_trn)[:,-1]
   score_tst = clf.predict_proba(X_tst)[:,-1]
   score_val = clf.predict_proba(X_val)[:,-1]
   
   # ROC curves
   roc_curves(dict={'trn':{'label':y_trn,'score':score_trn,'weight':trn.weight},
                    'tst':{'label':y_tst,'score':score_tst,'weight':tst.weight},
                    'val':{'label':y_val,'score':score_val,'weight':val.weight}},
              title='weights_roc_post')

   return df

