import numpy as np
import matplotlib.pyplot as plt

################################################################################
# 
def efficiency1( output_base,
                 value_total = np.random.exponential(scale=3, size=2000),
                 weight_total = 1.,
                 value_passed = np.random.normal(loc=6, scale=1, size=100),
                 weight_passed = 1., ) :
   
   bin_edges = np.linspace(0., 10., 40)
   # bin_edges = np.append( np.linspace(0., 4., 8, endpoint=False),
   #                       np.linspace(4., 11., 8, endpoint=True) )

   bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

   his_passed = np.histogram(value_passed,bins=bin_edges)[0]*weight_passed
   his_total = np.histogram(value_total,bins=bin_edges)[0]*weight_total
   his_total -= his_passed
   
   # Usually sumw2 would be part of the histogram filling package
   sumw2_total = his_total*weight_total
   sumw2_passed = his_passed*weight_passed

   #####

   fig_style = {
      'figsize': (5, 5),
      }

   ratio_fig_style = {
      'figsize': (5, 5),
      'gridspec_kw': {'height_ratios': (2, 2)},
      }
   
   stack_style = {
      'edgecolor': (0, 0, 0, 0.5),
      }
   
   hatch_style = {
      'facecolor': 'none',
      'edgecolor': (0, 0, 0, 0.5),
      'linewidth': 0,
      'hatch': '///',
      }
   
   errorbar_style = {
      'linestyle': 'none',
      'marker': '.',
      'elinewidth': 1,
      'capsize': 1,
      'color': 'k',
      }
   
   shaded_style = {
      'facecolor': (0,0,0,0.3),
      'linewidth': 0
      }
   
   #####

   fig, (ax, rax) = plt.subplots(2, 1, sharex=True, **ratio_fig_style)
   fig.subplots_adjust(hspace=.07)  # this controls the margin between the two axes

#   # Bkgd, test
#   yerr_low  = [ poisson_interval(n)[0]/his_bkgd_test.sum() for n in his_bkgd_test ]
#   yerr_high = [ poisson_interval(n)[1]/his_bkgd_test.sum() for n in his_bkgd_test ]
#   bkgd_test = ax.errorbar(x=bin_centres, 
#                           y=his_bkgd_test/his_bkgd_test.sum(), 
#                           yerr=np.sqrt(his_bkgd_test)/his_bkgd_test.sum(),
#                           label="Bkgd (test)",
#                           color=bkgd_train.get_color(),
#                           **errorbar_style
#                           )
#
#   (statistic,pvalue) = ks(his_bkgd_train/his_bkgd_train.sum(), 
#                           his_bkgd_test/his_bkgd_test.sum())
#   bkgd_test.set_label( bkgd_test.get_label() + ", KS p-value: {:4.2f}".format(pvalue) )

####

   # Stack up the various contributions
   labels = ['Passed','Total']
   sumw_stack = np.vstack([his_passed,his_total])
   # depending on step option ('pre' or 'post'), the last bin
   # needs be concatenated on one side, so that the edge bin is drawn
   sumw_stack = np.hstack([sumw_stack, sumw_stack[:,-1:]])
   ax.stackplot(bin_edges, sumw_stack, labels=labels, step='post', **stack_style)
   
   # Overlay an uncertainty hatch
   sumw_total = sumw_stack.sum(axis=0)
   unc = np.sqrt(sumw2_total)
   unc = np.hstack([unc, unc[-1]])
   ax.fill_between(x=bin_edges, y1=sumw_total - unc, y2=sumw_total + unc,
                   label='Stat. Unc.', step='post', **hatch_style
                   )

   #ax.set_yscale('log')
   ax.set_ylim(0., None)
   ax.set_ylabel('Counts')
   ax.legend()

   # Draw some sort of ratio, keeping the two uncertainty sources
   # separate rather than combining (as per tradition)
   #rax.fill_between(x=bin_edges, y1=1 - unc/sumw_total, y2=1 + unc/sumw_total, step='post', **shaded_style)
   sumw_total = sumw_total[:-1]  # Take away that copy of the last bin
   rax.errorbar(x=bin_centres, 
                y=[ x/y if y > 0 else -0.1 for x,y in zip(his_passed,sumw_total) ],
                yerr=[ np.sqrt(x)/y if y > 0 else 0. for x,y in zip(his_passed,sumw_total) ],
                **errorbar_style)
   
   rax.set_ylim(0., 1.)
   rax.set_ylabel('Efficiency')
   rax.set_xlabel('Transverse momentum (GeV)')
   rax.autoscale(axis='x', tight=True)

   plt.tight_layout()
   try : plt.savefig('{:s}/efficiency1.pdf'.format(output_base))
   except : print('Issue: {:s}/efficiency1.pdf'.format(output_base))
   plt.clf()
