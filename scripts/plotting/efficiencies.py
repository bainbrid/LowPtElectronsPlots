from . import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
#from matplotlib import rc
#rc('text',usetex=True)

################################################################################
# 
def efficiencies(path="./",
                 suffix="",
                 mistag=False,
                 title="test",
                 histograms=False,
                 curves={"test":{"legend":"test",
                                 "var":None,
                                 "mask":None,
                                 "condition":None}},
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

   #####

   ratio_fig_style = {
      'figsize': (5, 5),
      'gridspec_kw': {'height_ratios': (2, 2)},
      }
   
   errorbar_style = {
      'linestyle': 'none',
      'marker': '.',
      'elinewidth': 1,
      'capsize': 1,
      }

   if histograms :
      fig, (ax, rax) = plt.subplots(2, 1, sharex=True, **ratio_fig_style)
      fig.subplots_adjust(hspace=.07)  # this controls the margin between the two axes
   else :
      fig, rax = plt.subplots()

   #####

   for idx,(key,curve) in enumerate(curves.items()) :
      
      his_total = None
      his_passed = None
      if curve["var"] is None and curve["mask"] is None and curve["condition"] is None :
         his_total,_ = np.histogram(np.random.exponential(scale=2, size=10000),bins=bin_edges)
         his_passed,_ = np.histogram(np.random.normal(loc=2, scale=0.5, size=10000),bins=bin_edges)
         his_passed = np.cumsum(his_passed)
         his_passed = np.true_divide(his_passed,his_passed[-1])
         his_passed *= his_total
      else :
         his_total,_ = np.histogram(curve["var"][curve["mask"]],bins=bin_edges)
         his_passed,_ = np.histogram(curve["var"][curve["mask"]&curve["condition"]],bins=bin_edges)

      sum_passed = his_passed.sum()
      sum_total = his_total.sum()

      #print("key:  ",key)
      eff = sum_passed/sum_total
      #print("bins=  "," "*9,                           ", ".join(["{:6.1f}".format(x) for x in bin_edges]))
      #print("total= ","{:7.1f}".format(sum_total),":", ", ".join(["{:6.1f}".format(x) for x in his_total]))
      #print("passed=","{:7.1f}".format(sum_passed),":",", ".join(["{:6.1f}".format(x) for x in his_passed]))
      #print("eff=   ","{:7.2f}".format(eff),":",       ", ".join(["{:6.2f}".format(x/y if y > 0 else 0.) for x,y in zip(his_passed,his_total)]))
      
      his_passed = his_passed.astype(np.float64) / bin_widths
      his_total = his_total.astype(np.float64) / bin_widths

      if histograms :
         # Denominator
         denom, = ax.step(x=bin_edges,
                          y=np.hstack([his_total,his_total[-1]]),
                          where='post',
                          #linewidth=len(curves.keys())-idx,
                          color = curve["color"] if curve["color"] is not None else None,
                          )
         # Numerator
         yerr_low  = [ poisson_interval(int(n))[0] for n in his_passed ]
         yerr_high = [ poisson_interval(int(n))[1] for n in his_passed ]
         numer = ax.errorbar(x=bin_centres, 
                             y=his_passed, 
                             yerr=np.sqrt(his_passed),
                             label=curve["legend"],
                             color=denom.get_color(),
                             **errorbar_style
                             )

      # Efficiency
      rax.errorbar(x=bin_centres, 
                   y=[ x/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ],
                   yerr=[ np.sqrt(x)/y if y > 0 else 0. for x,y in zip(his_passed,his_total) ],
                   color=denom.get_color() if histograms else None,
                   label='{:5.3f}'.format(sum_passed*1./sum_total) if histograms else curve["legend"],
                   **errorbar_style)
      
   # Formatting for top panel
   if title is not "" : 
      if histograms : ax.set_title(title)
      else : rax.set_title(title)

   if histograms :
      ax.set_xlim(bin_edges[0],bin_edges[-1])
      ax.set_ylim(0.,None)
      ax.set_ylabel('Counts / {:.1f} GeV'.format(bin_width))

      new_handles = []
      new_labels = []
      new_handles.append(mlines.Line2D([],[], color='black', marker=None))
      new_labels.append("Denominator")
      new_handles.append(ax.errorbar([], [], yerr=0.5, color='black', **errorbar_style))
      new_labels.append("Numerator")
      handles,labels = ax.get_legend_handles_labels()
      for h,l in zip(handles,labels) :
         new_handles.append(mpatches.Patch(color=h.lines[0].get_color()))
         new_labels.append(l)
      ax.legend(handles=new_handles,labels=new_labels,frameon=False)
   
   # Formatting for bottom panel
   # rax.set_xlim(0., 10.)
   if mistag == True :
      rax.set_yscale('log') 
      rax.set_ylim(0.001, 1.)
      rax.set_ylabel('Mistag rate')
      #rax.set_ylabel('Mistag rate (mean={:5.3f})'.format(sum_passed*1./sum_total))
   else :
      rax.set_yscale('linear')
      rax.set_ylim(0., 1.)
      rax.set_ylabel('Efficiency')
      #rax.set_ylabel('Efficiency (mean={:4.2f})'.format(sum_passed*1./sum_total))
   rax.set_xlabel('Transverse momentum (GeV)')
   rax.autoscale(axis='x', tight=True)

   if histograms :
      new_handles = []
      new_labels = []
      new_handles.append(mlines.Line2D([],[], color='white', marker=None))
      new_labels.append("Mean:")
      handles,labels = rax.get_legend_handles_labels()
      for h,l in zip(handles,labels) :
         new_handles.append(h)
         new_labels.append(l)
      rax.legend(handles=new_handles,labels=new_labels,frameon=False)
   else :
      rax.legend()

   plt.tight_layout()
   name = "mistag_rates" if mistag == True else "efficiencies"
   try : plt.savefig(f'{path}/{name}{suffix}.pdf'.format(path,name,suffix))
   except : print(f'Issue: {path}/{name}{suffix}.pdf'.format(path,name,suffix))
   plt.clf()




   #####
   #####
   #####
   #####

#   fig_style = {
#      'figsize': (5, 5),
#      }
#
#   ratio_fig_style = {
#      'figsize': (5, 5),
#      'gridspec_kw': {'height_ratios': (2, 2)},
#      }
#   
#   stack_style = {
#      'edgecolor': (0, 0, 0, 0.5),
#      }
#   
#   hatch_style = {
#      'facecolor': 'none',
#      'edgecolor': (0, 0, 0, 0.5),
#      'linewidth': 0,
#      'hatch': '///',
#      }
#   
#   errorbar_style = {
#      'linestyle': 'none',
#      'marker': '.',
#      'elinewidth': 1,
#      'capsize': 1,
#      'color': 'k',
#      }
#   
#   shaded_style = {
#      'facecolor': (0,0,0,0.3),
#      'linewidth': 0
#      }
   
#   his_total = None
#   his_passed = None
#   if curve["var"] is None and curve["mask"] is None and curve["condition"] is None :
#      his_total,_ = np.histogram(np.random.exponential(scale=2, size=10000),bins=bin_edges)
#      his_passed,_ = np.histogram(np.random.normal(loc=2, scale=0.5, size=10000),bins=bin_edges)
#      his_passed = np.cumsum(his_passed)
#      his_passed = np.true_divide(his_passed,his_passed[-1])
#      his_passed *= his_total
#   else :
#      his_passed,_ = np.histogram(value_passed,bins=bin_edges)
#      his_total,_ = np.histogram(value_total,bins=bin_edges)
#
#   sum_passed = his_passed.sum()
#   sum_total = his_total.sum()
#   
#   his_passed = his_passed.astype(np.float64) / bin_widths
#   his_total = his_total.astype(np.float64) / bin_widths
#   his_total -= his_passed
   
   #####
   
#   fig, (ax, rax) = plt.subplots(2, 1, sharex=True, **ratio_fig_style)
#   fig.subplots_adjust(hspace=.07)  # this controls the margin between the two axes
#
#   # Stack up the various contributions
#   labels = [f'Passed ({sum_passed})'.format(sum_passed),f'Total ({sum_total})'.format(sum_total)]
#   sumw_stack = np.vstack([his_passed,his_total])
#   # depending on step option ('pre' or 'post'), the last bin
#   # needs be concatenated on one side, so that the edge bin is drawn
#   sumw_stack = np.hstack([sumw_stack, sumw_stack[:,-1:]])
#   ax.stackplot(bin_edges, sumw_stack, labels=labels, step='post', **stack_style)
#   
#   # Overlay an uncertainty hatch
#   sumw_total = sumw_stack.sum(axis=0)
#   unc = np.sqrt(his_total)
#   unc = np.hstack([unc, unc[-1]])
#   ax.fill_between(x=bin_edges, y1=sumw_total - unc, y2=sumw_total + unc,
#                   label='Stat. Unc.', step='post', **hatch_style
#                   )
#
#   if title is not "" : ax.set_title(title)
#   ax.set_xlim(0., 10.)
#   if mistag == True :
#      ax.set_yscale('log')
#      ax.set_ylim(0.5, None)
#   else :
#      ax.set_yscale('linear')
#      ax.set_ylim(0., None)
#   ax.set_ylabel(f'Counts / {bin_width} GeV'.format(bin_width))
#   ax.legend()
