import os
from glob import glob
from argparse import ArgumentParser

# python eval_bdt.py $(pwd)/models/2019Jul22/bdt_cmssw_mva_id

parser = ArgumentParser()
parser.add_argument('model', default='$(pwd)/models/2019Jul22/bdt_cmssw_mva_id')
args = parser.parse_args()

pkls = glob('%s/*.pkl' % args.model)
if len(pkls) != 1 : raise RuntimeError('There must be one and only one pkl in the directory')
base = os.path.basename(args.model)
what = base[4:]
model = pkls[0]

#this should be outsorced
from features import *
features, additional = get_features(what)

print 'Loading model'
from sklearn.externals import joblib
import xgboost as xgb
xml = model.replace('.pkl', '.xml')
model = joblib.load(model)
from datasets import HistWeighter

from xgbo.xgboost2tmva import convert_model
from itertools import cycle

# xgb sklearn API assigns default names to the variables, use that to dump the XML
# then convert them to the proper name
print 'XML conversion'
xgb_feats = ['f%d' % i for i in range(len(features))]
convert_model(model._Booster.get_dump(), zip(xgb_feats, cycle('F')), xml)
xml_str = open(xml).read()
for idx, feat in reversed(list(enumerate(features))):
    xml_str = xml_str.replace('f%d' % idx, feat)
with open(xml.replace('.xml', '.fixed.xml'), 'w') as XML:
    XML.write(xml_str)
