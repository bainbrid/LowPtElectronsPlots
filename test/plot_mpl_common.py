import matplotlib.pyplot as plt

################################################################################

def draw_mpl_wp(eff,fr,**kwargs):
    plt.plot(
        [fr],
        [eff],
        marker='o',
        markerfacecolor=kwargs.get('mfc','black'),
        markeredgecolor=kwargs.get('mec',kwargs.get('mfc','black')),
        markersize=kwargs.get('markersize',8),
        linestyle='none',
        label=kwargs.get('label','unknown'),
        )

################################################################################

def draw_mpl_roc(tpr,fpr,auc,eff=1.,fr=1.,**kwargs):
    label = kwargs.get('label','unknown')
    if auc is not None: label += ' ({:.3f})'.format(auc)
    plt.plot(
        fpr*fr,
        tpr*eff,
        linestyle=kwargs.get('linestyle','solid'),
        color=kwargs.get('color','black'),
        linewidth=1.0,
        label=label)

################################################################################
