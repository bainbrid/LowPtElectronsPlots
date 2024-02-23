from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rc
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
matplotlib.use('Agg')
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def debug(df,str=None,is_egamma=False) :
   
   if str is not None :
      print str
   elif is_egamma :
      print "EGAMMA"
   else :
      print "LOW PT"

   has_trk = (df.has_trk) & (df.trk_pt>0.5) & (np.abs(df.trk_eta)<2.4)
   if is_egamma :
      has_gsf = (df.has_pfgsf) & (df.pfgsf_pt>0.5) & (np.abs(df.pfgsf_eta)<2.4)
   else :
      has_gsf = (df.has_gsf) & (df.gsf_pt>0.5) & (np.abs(df.gsf_eta)<2.4)
   has_ele = (df.has_ele) & (df.ele_pt>0.5) & (np.abs(df.ele_eta)<2.4)
   print pd.crosstab(df.is_e,
                     [has_trk,has_gsf,has_ele],
                     rownames=['is_e'],
                     colnames=['has_trk','has_pfgsf' if is_egamma else 'has_gsf','has_ele'],
                     margins=True)

################################################################################
# method that handles nans to provide backward compatibility w.r.t. mauro's ntuples
def backward(data) :

   # replicate 'nan' values in old ntuples
   data.replace(-10.,-1.,inplace=True)
   data.replace(-10,-1,inplace=True)
   variables = [x for x in data.columns if x.startswith('eid_')]
   data[variables].replace(-10.,-666.,inplace=True)

   # replace -10. with -1. for 
   variables = ["trk_p","trk_chi2red","gsf_chi2red","sc_E","sc_eta","sc_etaWidth",
                "sc_phiWidth","match_seed_dEta","match_eclu_EoverP","match_SC_EoverP",
                "match_SC_dEta","match_SC_dPhi","shape_full5x5_sigmaIetaIeta",
                "shape_full5x5_sigmaIphiIphi","shape_full5x5_HoverE","shape_full5x5_r9",
                "shape_full5x5_circularity","rho","brem_frac","ele_pt",]
   data[variables].replace(-10.,-1.,inplace=True)
   data[["trk_nhits","gsf_nhits"]].replace(-10,-1,inplace=True)

   # assume modified df is not pass-by-reference?
   return data

################################################################################
# method to add ROC curve to plot 
def plot( plt, df, string, selection, draw_roc, draw_eff, 
          label, color, 
          markerstyle, markersize, linestyle, linewidth=1.0, discriminator=None, mask=None, 
          df_xaxis=None ) :
   
   if draw_roc is True and discriminator is None : 
      print "No discriminator given for ROC curve!"
      quit()
   print string
   if mask is None : mask = [True]*df.shape[0]
   denom = df.is_e#[mask]; 
   numer = denom & selection#[mask]
   eff = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   print "   eff/numer/denom: {:6.4f}".format(eff), numer.sum(), denom.sum()
   denom = ~df.is_e[mask]; numer = denom & selection[mask]
   if df_xaxis is not None : denom = ~df_xaxis.is_e # change x-axis denominator!
   mistag = float(numer.sum()) / float(denom.sum()) if float(denom.sum()) > 0. else 0.
   print "    fr/numer/denom: {:6.4f}".format(mistag), numer.sum(), denom.sum()
   if draw_roc :
      roc = roc_curve(df.is_e[selection&mask], discriminator[selection&mask])
      auc = roc_auc_score(df.is_e[selection&mask], discriminator[selection&mask])
      plt.plot(roc[0]*mistag,
               roc[1]*eff,
               linestyle=linestyle,
               linewidth=linewidth,
               color=color)#+', AUC: %.3f'%auc)
      plt.plot([mistag], [eff], 
               marker=markerstyle, 
               markersize=markersize,
               linestyle=linestyle,
               color=color,
               label=label)
   elif draw_eff :
      plt.plot([mistag], [eff], 
               marker=markerstyle, 
               markersize=markersize, 
               linestyle='',
               color=color, 
               label=label)
   if draw_roc :
      return eff,mistag,roc
   if draw_eff :
      return eff,mistag,None

################################################################################
# import all methods used below 
from plotting_methods_new import *

################################################################################
# top-level wrap to produce plots

def plotting(plots,dataset,args,df_lowpt,df_egamma,df_orig) :
   print "plotting() ..."

   plots_list = [
      {"method":AxE_retraining,"args":(plt,df_lowpt,df_egamma),"suffix":"AxE_retraining",},
      ]
   
   for plot in plots_list :

      plt.figure()
      ax = plt.subplot(111)
      box = ax.get_position()
      ax.set_position([box.x0, box.y0, box.width, box.height*0.666])
      #plt.title('%s training' % args.what.replace("_"," "))
      plt.plot(np.arange(0.,1.,0.01),np.arange(0.,1.,0.01),'k--')

      ax.tick_params(axis='x', pad=10.)
      #ax.text(0, 1, r'\bf{CMS}\ \it{Simulation}\ \it{Preliminary}', 
      ax.text(0, 1, '\\textbf{CMS} \\textit{Simulation} \\textit{Preliminary}', 
              ha='left', va='bottom', transform=ax.transAxes)
      ax.text(1, 1, r'13 TeV', 
              ha='right', va='bottom', transform=ax.transAxes)

      plot["method"](*plot["args"]) # Execute method
 
      # Adapt legend
      def update_prop(handle, orig):
         handle.update_from(orig)
         #handle.set_marker("o")
      plt.legend(handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)})

      plt.xlabel('Mistag rate')
      plt.ylabel(r'Acceptance $\times$ efficiency')
      #plt.legend(loc='lower left', bbox_to_anchor=(0., 1.1)) #plt.legend(loc='best')
      #plt.xlim(0., 1)
      
      tupl = (plots, dataset, args.jobtag, args.what, plot["suffix"])
      #try : plt.savefig('%s/%s_%s_%s_BDT_%s.png' % (tupl))
      #except : print 'Issue: %s/%s_%s_%s_BDT_%s.png' % (tupl)
      #try : plt.savefig('%s/%s_%s_%s_BDT_%s.pdf' % (tupl))
      #except : print 'Issue: %s/%s_%s_%s_BDT_%s.pdf' % (tupl)
      plt.gca().set_xscale('log')
      plt.xlim(1e-4, 1)
      try : plt.savefig('%s/%s_%s_%s_log_BDT_%s.png' % (tupl))
      except : print 'Issue: %s/%s_%s_%s_log_BDT_%s.png' % (tupl)
      try : plt.savefig('%s/%s_%s_%s_log_BDT_%s.pdf' % (tupl))
      except : print 'Issue: %s/%s_%s_%s_log_BDT_%s.pdf' % (tupl)
      
      plt.clf()
