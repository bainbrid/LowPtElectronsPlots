import numpy as np
import matplotlib.pyplot as plt

################################################################################
# 
def efficiency( output_base="./",
                mistag=False,
                title="",
                suffix="",
                value_total=None,
                value_passed=None,
                ) :

   #bin_edges = np.linspace(0., 10., 41, endpoint=True)
   bin_edges = np.linspace(0., 4., 8, endpoint=False)
   bin_edges = np.append( bin_edges, np.linspace(4., 8., 4, endpoint=False) )
   bin_edges = np.append( bin_edges, np.linspace(8., 10., 2, endpoint=True) )

   bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
   bin_widths = (bin_edges[1:] - bin_edges[:-1])
   bin_width = bin_widths[0]
   bin_widths /= bin_width
   #print(bin_edges)
   #print(bin_centres)
   #print(bin_width)
   #print(bin_widths)
   
   his_total = None
   his_passed = None
   if value_total is None and value_passed is None :
      his_total,_ = np.histogram(np.random.exponential(scale=2, size=10000),bins=bin_edges)
      his_passed,_ = np.histogram(np.random.normal(loc=2, scale=0.5, size=10000),bins=bin_edges)
      his_passed = np.cumsum(his_passed)
      his_passed = np.true_divide(his_passed,his_passed[-1])
      his_passed *= his_total
   else :
      his_passed,_ = np.histogram(value_passed,bins=bin_edges)
      his_total,_ = np.histogram(value_total,bins=bin_edges)

   sum_passed = his_passed.sum()
   sum_total = his_total.sum()

   his_passed = his_passed.astype(np.float64) / bin_widths
   his_total = his_total.astype(np.float64) / bin_widths
   his_total -= his_passed

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

   # Stack up the various contributions
   labels = [f'Passed ({sum_passed:.0f})'.format(sum_passed),f'Total ({sum_total:.0f})'.format(sum_total)]
   sumw_stack = np.vstack([his_passed,his_total])
   # depending on step option ('pre' or 'post'), the last bin
   # needs be concatenated on one side, so that the edge bin is drawn
   sumw_stack = np.hstack([sumw_stack, sumw_stack[:,-1:]])
   ax.stackplot(bin_edges, sumw_stack, labels=labels, step='post', **stack_style)
   
   # Overlay an uncertainty hatch
   sumw_total = sumw_stack.sum(axis=0)
   unc = np.sqrt(his_total)
   unc = np.hstack([unc, unc[-1]])
   ax.fill_between(x=bin_edges, y1=sumw_total - unc, y2=sumw_total + unc,
                   label='Stat. Unc.', step='post', **hatch_style
                   )

   if title is not "" : ax.set_title(title)
   ax.set_xlim(0., 10.)
   if mistag == True :
      ax.set_yscale('log')
      ax.set_ylim(0.5, None)
   else :
      ax.set_yscale('linear')
      ax.set_ylim(0., None)
   ax.set_ylabel(f'Counts / {bin_width} GeV'.format(bin_width))
   ax.legend()

   sumw_total = sumw_total[:-1]  # Take away that copy of the last bin
   rax.errorbar(x=bin_centres, 
                y=[ x/y if y > 0 else -0.1 for x,y in zip(his_passed,sumw_total) ],
                yerr=[ np.sqrt(x)/y if y > 0 else 0. for x,y in zip(his_passed,sumw_total) ],
                **errorbar_style)
   
   rax.set_xlim(0., 10.)
   if mistag == True :
      rax.set_yscale('log') 
      rax.set_ylim(0.001, 1.)
      rax.set_ylabel('Mistag rate (mean={:5.3f})'.format(sum_passed*1./sum_total))
   else :
      rax.set_yscale('linear')
      rax.set_ylim(0., 1.)
      rax.set_ylabel('Efficiency (mean={:4.2f})'.format(sum_passed*1./sum_total))
   rax.set_xlabel('Transverse momentum (GeV)')
   rax.autoscale(axis='x', tight=True)

   plt.tight_layout()
   name = "mistag" if mistag == True else "efficiency"
   try : plt.savefig(f'{output_base}/{name}{suffix}.pdf'.format(output_base,name,suffix))
   except : print(f'Issue: {output_base}/{name}{suffix}.pdf'.format(output_base,name,suffix))
   plt.clf()
