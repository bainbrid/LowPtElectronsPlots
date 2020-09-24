import numpy as np

################################################################################
# Poisson intervals

poisson = [
   (0,1.84102),
   (0,1.84102),
   (0.827246,2.29953),
   (1.29181,2.63786),
   (1.6327,2.91819),
   (1.91434,3.16275),
   (2.15969,3.38247),
   (2.37993,3.58364),
   (2.58147,3.77028),
   (2.76839,3.94514),
   ]
def poisson_uncertainty(n) : return poisson[n] if n < 9 else (np.sqrt(n),np.sqrt(n))

from scipy.stats import gamma
def poisson_interval(mu,cl=0.683) :
    return ( mu-gamma.interval(cl,mu)[0] if mu > 0. else 0., 
             gamma.interval(cl,mu+1.)[1]-mu )

if __name__ == '__main__' :
   print(" ".join([ "{0:1.0f}+{2:4.2f}-{1:4.2f}".format(n,np.sqrt(n),np.sqrt(n)) for n in range(0,10) ]))
   print(" ".join([ "{0:1.0f}+{2:4.2f}-{1:4.2f}".format(n,*poisson_interval(n)) for n in range(0,10) ]))
   print(" ".join([ "{0:1.0f}+{2:4.2f}-{1:4.2f}".format(n,*poisson_interval(n)) for n in range(0,10) ]))
