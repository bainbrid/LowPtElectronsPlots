import pandas as pd
import uproot
from config import *

################################################################################

def get_data(files,nevents=-1) :

    print('Getting files:\n', '\n'.join(files),'...')
    dfs = [ uproot.open(i)['ntuplizer/tree'].arrays(columns,library="pd")  for i in files ]
    df = pd.concat(dfs)
    print('Done!')

    if verbosity > 0:
        print('Available branches: ',df.keys())
        print('Features for model: ',features)

    if nevents > 0 : 
        print("Considering only first {:.0f} events ...".format(nevents))
        df = df.head(nevents)
    return df

#def print_branches(files):
#    print('Getting file:',f)
#    df = uproot.open(f)['ntuplizer/tree']
#    branches = df.keys()
#    print('All available branches:\n',branches)
#    print()
#    print('Seeding-related branches:\n',[b for b in branches if 'preid' in b])
#    print()
