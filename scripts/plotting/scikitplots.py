from __future__ import print_function
from __future__ import absolute_import
import builtins
import future
from future.utils import raise_with_traceback
import past
import six

import matplotlib
matplotlib.use('Agg') # choose backend before doing anything else with pyplot! ('macosx')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties

import numpy as np
import scikitplot as skplt
from scipy.special import softmax
print("version:",skplt.__version__)

################################################################################
# 
def scikitplots(test,egamma) :
   
   dir = "temp/scikitplots"

   y_test = test.is_e
   #probas = test['training_out']
   probas = test['gsf_bdtout1']

   has_gsf = (test.has_gsf) & (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.4)
   has_ele = (test.has_ele) & (test.ele_pt>0.5) & (np.abs(test.ele_eta)<2.4)
   #mask = (test.trk_pt>0.5) & (np.abs(test.trk_eta)<2.4) & (test.ele_pt>0.) 
   #mask = (test.gsf_pt>0.5) & (np.abs(test.gsf_eta)<2.4) & (test.ele_pt>0.) 
   mask = has_gsf
   
   y_test = y_test.to_numpy()[mask]
   probas = np.reshape(probas.to_numpy(),(-1,1))[mask]
   probas = np.hstack((-probas,probas))
   probas = softmax(probas, axis=1)
   
   #np.set_printoptions(precision=2)
   #print(type(y_test))
   #print(type(probas))
   #print(y_test.shape)
   #print(probas.shape)
   #print(y_test[:20])
   #print(probas[:20])
   #print(probas_[:20])
    
   # https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
   skplt.metrics.plot_calibration_curve(y_test,probas_list=[probas],clf_names=['XGBoost'],n_bins=10)
   plt.savefig(dir+'/plot_calibration_curve.pdf')
   
   # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
   skplt.metrics.plot_confusion_matrix(y_test,probas[:,1]>0.5,normalize=True)
   plt.savefig(dir+'/plot_confusion_matrix.pdf')
   
   skplt.metrics.plot_cumulative_gain(y_test,probas)
   plt.savefig(dir+'/plot_cumulative_gain.pdf')
   
#   # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#   if args.train :
#      skplt.estimators.plot_feature_importances(model)
#      plt.savefig(dir+'/feature_importances.pdf')
   
   skplt.metrics.plot_ks_statistic(y_test,probas)
   plt.savefig(dir+'/plot_ks_statistic.pdf')
   
#   # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#   if args.train :
#      skplt.estimators.plot_learning_curve(clf,train[features].values,train.is_e.values )
#      plt.savefig(dir+'/plot_learning_curve.pdf')
   
   skplt.metrics.plot_lift_curve(y_test,probas)
   plt.savefig(dir+'/plot_lift_curve.pdf')
   
   # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
   skplt.metrics.plot_precision_recall(y_test,probas)
   plt.savefig(dir+'/plot_precision_recall.pdf')
   
   # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
   plt.figure()
   ax = plt.subplot(111)
   skplt.metrics.plot_roc(y_test,probas, plot_micro=False, plot_macro=False, ax=ax, classes_to_plot=[0])
   ax.lines.remove(ax.lines[-1]) # hack: remove "by chance" line, broken by logx
   skplt.metrics.plot_roc(y_test,probas, plot_micro=False, plot_macro=False, ax=ax, classes_to_plot=[1])
   ax.lines.remove(ax.lines[-1]) # hack: remove "by chance" line, broken by logx
   plt.title("")
   plt.xlim(1.e-4,1.)
   plt.gca().set_xscale('log')
   plt.plot(np.arange(0.,1.,plt.xlim()[0]),np.arange(0.,1.,plt.xlim()[0]),'k--')
   plt.legend(loc='best')
   plt.tight_layout()
   plt.savefig(dir+'/plot_roc.pdf')
