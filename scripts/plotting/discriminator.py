import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp as ks
from . import *

################################################################################
# Plot BDT discriminator distributions for signal and bkgd from train and test samples
def discriminator( output_base,
                   title = "",
                   suffix = "",
                   signal_train = np.random.normal(loc=3, scale=1, size=10000),
                   signal_test = np.random.normal(loc=3, scale=1, size=1000),
                   bkgd_train = np.random.normal(loc=0, scale=1, size=10000),
                   bkgd_test = np.random.normal(loc=0, scale=1, size=1000)
                   ) :
   
   bin_edges = np.linspace(-10.,12.,45,endpoint=True)
   bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
   #print("bin_edges",bin_edges)
   
   his_signal_train = np.histogram(signal_train,bins=bin_edges)[0]
   his_signal_test = np.histogram(signal_test,bins=bin_edges)[0]
   his_bkgd_train = np.histogram(bkgd_train,bins=bin_edges)[0]
   his_bkgd_test = np.histogram(bkgd_test,bins=bin_edges)[0]

   ratio_fig_style = {
      'figsize': (5, 5),
      'gridspec_kw': {'height_ratios': (3, 1)},
      }
   
   errorbar_style = {
      'linestyle': 'none',
      'marker': '.',
      'elinewidth': 1,
      'capsize': 1,
      }

   fig, (ax, rax) = plt.subplots(2, 1, sharex=True, **ratio_fig_style)
   fig.subplots_adjust(hspace=.1)  # this controls the margin between the two axes

   # Bkgd, train
   bkgd_train, = ax.step(x=bin_edges,
                         y=np.hstack([his_bkgd_train/his_bkgd_train.sum(), 
                                      (his_bkgd_train/his_bkgd_train.sum())[-1]]),
                         where='post',
                         label="Bkgd (train)",
                         alpha=0.5
                         )

   # Bkgd, test
   yerr_low  = [ poisson_interval(n)[0]/his_bkgd_test.sum() for n in his_bkgd_test ]
   yerr_high = [ poisson_interval(n)[1]/his_bkgd_test.sum() for n in his_bkgd_test ]
   bkgd_test = ax.errorbar(x=bin_centres, 
                           y=his_bkgd_test/his_bkgd_test.sum(), 
                           yerr=np.sqrt(his_bkgd_test)/his_bkgd_test.sum(),
                           label="Bkgd (test)",
                           color=bkgd_train.get_color(),
                           **errorbar_style
                           )

   (statistic,pvalue) = ks(his_bkgd_train/his_bkgd_train.sum(), 
                           his_bkgd_test/his_bkgd_test.sum())
   bkgd_test.set_label( bkgd_test.get_label() + ", KS p-value: {:4.2f}".format(pvalue) )

   # Signal, train
   signal_train, = ax.step(x=bin_edges,
                           y=np.hstack([his_signal_train/his_signal_train.sum(), 
                                        (his_signal_train/his_signal_train.sum())[-1]]),
                           where='post',
                           label="Signal (train)",
                           alpha=0.5
                           )

   # Signal, test
   y_values  = his_signal_test/his_signal_test.sum()
   yerr_low  = [ poisson_interval(n)[0]/his_signal_test.sum() for n in his_signal_test ]
   yerr_high = [ poisson_interval(n)[1]/his_signal_test.sum() for n in his_signal_test ]
   signal_test = ax.errorbar(x=bin_centres, 
                             y=y_values, 
                             yerr=[yerr_low,yerr_high],
                             label="Signal (test)",
                             color=signal_train.get_color(),
                             **errorbar_style
                             )
   (statistic,pvalue) = ks(his_signal_train/his_signal_train.sum(), 
                           his_signal_test/his_signal_test.sum())
   signal_test.set_label( signal_test.get_label() + ", KS p-value: {:4.2f}".format(pvalue) )
   print("signal KS:",statistic,pvalue)

   if title is not "" : ax.set_title(title)
   ax.set_yscale('log')
   ax.set_ylim(0.001,1.)
   ax.set_ylabel('a.u.')

   indices = []
   handles,labels = ax.get_legend_handles_labels()
   for label in ["Signal (test)","Bkgd (test)","Signal (train)","Bkgd (train)"] :
      for i,elem in enumerate(labels) :
         if label in elem: indices.append(i)
   new_labels = [ labels[i] for i in indices ]
   new_handles = [ handles[i] for i in indices ]
   ax.legend(new_handles,new_labels,frameon=False)

   # Ratio plot (bkgd)

   plt.hlines([1.], rax.get_xlim()[0], rax.get_xlim()[1], colors='grey', linewidth=0.5, linestyles='dashed', label='')

   y_values  = [ x/y if y > 0. else 0. for x,y in zip(his_bkgd_test/his_bkgd_test.sum(),
                                                      his_bkgd_train/his_bkgd_train.sum()) ]
   yerr_low  = [ r*poisson_interval(x)[0]/x if x > 0. else poisson_interval(x)[0]/1 for x,r in zip(his_bkgd_train,y_values) ]
   yerr_high = [ r*poisson_interval(x)[1]/x if x > 0. else poisson_interval(x)[1]/1 for x,r in zip(his_bkgd_train,y_values) ]
   rax.errorbar(x=bin_centres, 
                y=y_values,
                yerr=[yerr_low,yerr_high],
                **errorbar_style)

   # Ratio plot (signal)
   rax.errorbar(x=bin_centres, 
                y=[ x/y if y > 0. else -0.1 for x,y in zip(his_signal_test/his_signal_test.sum(),
                                                           his_signal_train/his_signal_train.sum()) ],
                yerr=[ np.sqrt(x)/x if x > 0. else 0. for x in his_signal_test ],
                **errorbar_style)

   rax.set_yscale('linear')
   rax.set_ylim(0.,3.)
   rax.set_ylabel('Test / Train')
   rax.set_xlabel('BDT discrimator value')
   rax.autoscale(axis='x',tight=True)

   plt.tight_layout()
   try : plt.savefig('{:s}/discriminator{:s}.pdf'.format(output_base,suffix))
   except : print('Issue: {:s}/discriminator{:s}.pdf'.format(output_base,suffix))
   plt.clf()
