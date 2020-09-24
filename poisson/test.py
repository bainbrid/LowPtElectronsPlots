import numpy as np
import matplotlib.pyplot as plt

def test_errorbar():
    x = np.arange(0.1, 4, 0.5)
    y = np.arange(0.1, 4, 0.5)#np.exp(-x)

    yerr = 0.5*x#0.1 + 0.2*np.sqrt(x)
    xerr = 0.1 + x#yerr

    # First illustrate basic pyplot interface, using defaults where possible.
    fig = plt.figure()
    ax = fig.gca()
    ax.errorbar(x, y, xerr=0.2, yerr=0.4)
    ax.set_title("Simplest errorbars, 0.2 in x, 0.4 in y")

    # Now switch to a more OO interface to exercise more features.
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0, 0]
    ax.errorbar(x, y, yerr=yerr, fmt='o')
    ax.set_title('Vert. symmetric')

    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)

    ax = axs[0, 1]
    ax.errorbar(x, y, xerr=xerr, fmt='o', alpha=0.4)
    ax.set_title('Hor. symmetric w/ alpha')

    ax = axs[1, 0]
    ax.errorbar(x, y, yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
    ax.set_title('H, V asymmetric')

    ax = axs[1, 1]
    ax.set_yscale('log')
    # Here we have to be careful to keep all y values positive:
    ylower = np.maximum(1e-2, y - yerr)
    yerr_lower = y - ylower

    ax.errorbar(x, y, yerr=[yerr_lower, 2*yerr], xerr=xerr,
                fmt='o', ecolor='g', capthick=2)
    ax.set_title('Mixed sym., log y')

    fig.suptitle('Variable errorbars') 
    try : plt.savefig('./test.pdf')
    except : print('Issue: ./test.pdf')

test_errorbar()
