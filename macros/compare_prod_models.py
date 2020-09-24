import numpy as np
import matplotlib
matplotlib.use('Agg')
from cmsjson import CMSJson
from pdb import set_trace
import os
from glob import glob
import pandas as pd
import json
from pprint import pprint
import matplotlib.pyplot as plt
from features import *
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.externals import joblib
import xgboost as xgb
from datasets import HistWeighter
from xgbo.xgboost2tmva import convert_model
from itertools import cycle
from sklearn.metrics import roc_curve

def get_model(pkl):
    model = joblib.load(pkl)

    def _monkey_patch():
        return model._Booster

    if isinstance(model.booster, basestring):
        model.booster = _monkey_patch
    return model


#
# Everything hardcoded because there is no reason to have a flexible approach
#

test = pd.read_hdf(
    'models/2018Nov01/bdt_bo_displaced_improvedfullseeding_noweight'
    '/nn_bo_displaced_improvedfullseeding_testdata.hdf', key='data')

biased = get_model(
    'models/2018Nov01/bdt_bo_displaced_improvedfullseeding_noweight'
    '/model_18.pkl')
biased_features, _ = get_features('displaced_improvedfullseeding')
test['biased_out'] = biased.predict_proba(test[biased_features].as_matrix())[:,1]
test['biased_out'].loc[np.isnan(test.biased_out)] = -999 #happens rarely, but happens
biased_roc = roc_curve(
    test.is_e, test.biased_out
    )

unbiased = get_model(
    'models/2018Nov01/bdt_bo_improvedfullseeding/'
    'model_24.pkl')
unbiased_features, _ = get_features('improvedfullseeding')
test['unbiased_out'] = unbiased.predict_proba(test[unbiased_features].as_matrix())[:,1]
test['unbiased_out'].loc[np.isnan(test.unbiased_out)] = -999 #happens rarely, but happens

print ''
jmap = {}
for biased_thr, unbiased_thr, wpname in [
    (1.83 , 2.61, 'T'),
    (0.76 , 1.75, 'M'),
    (-0.48, 1.03, 'L'),
    (1.45 , 2.61, 'T+'),
    (0.33 , 1.75, 'M+'),
    (-0.97, 1.03, 'L+'),
    ]:
    print 'WP', wpname
    test['biased_pass'] = test.biased_out > biased_thr
    test['unbiased_pass'] = test.unbiased_out > unbiased_thr
    test['or_pass'] = (test.biased_pass | test.unbiased_pass)
    
    eff = ((test.or_pass & test.is_e).sum()/float(test.is_e.sum()))
    mistag = ((test.or_pass & np.invert(test.is_e)).sum()/float(np.invert(test.is_e).sum()))

    jmap[wpname] = [mistag, eff]
    print 'OR wp'
    print 'eff: %.3f' % eff
    print 'mistag: %.3f' % mistag
    idx = np.abs(biased_roc[0] - mistag).argmin()
    print 'similar mistag: %.2f\t%.4f\t%.2f' % (biased_roc[1][idx], biased_roc[0][idx], biased_roc[2][idx])
    
    for mask, name in [
        (test.is_e, 'sig'),
        (np.invert(test.is_e), 'bkg')]:
        print name
        df = test[mask]
        pass_or = float(df.or_pass.sum())
        print 'pass both: %.3f' % ((df.biased_pass & df.unbiased_pass).sum()/pass_or)
        print 'pass bias only: %.3f' % ((df.biased_pass & np.invert(df.unbiased_pass)).sum()/pass_or)
        print 'pass unbias only: %.3f' % ((np.invert(df.biased_pass) & df.unbiased_pass).sum()/pass_or)

with open('model_or.json', 'w') as jfile:
    jfile.write(json.dumps(jmap))
