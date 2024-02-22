# LowPtElectronsPlots

## 'master' branch

Contains latest devs, aimed at EGamma paper.

## 'EXO-23-007' branch

Provides the performance plots for the scouting/parking review paper (EXO-23-007).

Recipe:
```
cd scripts
conda activate root6
python train2.py
# default input: ../data/170823/nonres_large/output_large.root
# default output: ../plots/
```

Input ntuples above are produced as follows:
```
# log into lxplus7
cd ~/work/public/00-obsolete/7-slc7/bparking/CMSSW_10_2_15/
cmsenv
cd src/LowPtElectrons/LowPtElectrons
# from repo: https://github.com/bainbrid/LowPtElectrons
cd run
# Interactive:
cmsRun ntuplizer_test_cfg.py # MC
cmsRun ntuplizer_data_cfg.py # data?
# Note above can be run with the option "simple=1"
#   default:  plugins/IDNtuplizer.cc
#   simple=1: plugins/IDNtuplizerSimple.cc
# CRAB:
crab submit crab_cfg.py
# Output ntuples here: /eos/user/b/bainbrid/lowpteleid/
```

## Other branches

- parking_paper: an old/obsolete version of the EXO-23-007 branch
- master_backup: old (unknown version of master, prior to merge of EXO-23-007)
- reproduce_mauro_rocs: as it says on the tin.
