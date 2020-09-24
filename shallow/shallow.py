import pandas as pd
import numpy as np

################################################################################
# CLI

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--nevents',default=-1,type=int)
parser.add_argument('--reweight',action='store_true')
parser.add_argument('--nbins',default=600,type=int)
args = parser.parse_args()
print("Command line args:",vars(args))

################################################################################
# Variables

from utils import features
from utils import additional
from utils import labelling
from utils import columns

if args.verbose :
   print('Features:',columns)
   print('Additional:',additional)
   print('Labelling:',labelling)

################################################################################
# Parse files

files = ['../data/output_prescale40_91k.root']
print('Input files:', ', '.join(["'{:s}'".format(f) for f in files]))

from utils import parse
df = parse(files,args.nevents,args.verbose)
print('df.shape:',df.shape)

################################################################################
# Preprocessing

from utils import preprocess
df = preprocess(df)

# print summary info
if args.verbose :
   print(df.info())
   with pd.option_context('display.width',None,
                          'display.max_rows',None, 
                          'display.max_columns',None,
                          'display.float_format','{:,.2f}'.format) :
      print(df.describe(include='all').T)

################################################################################
# Reweighting by pT,eta

from utils import calc_weights
df = calc_weights(df=df,
                  nbins=args.nbins,
                  reweight=args.reweight,
                  verbose=args.verbose)
   

