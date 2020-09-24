import numpy
from scipy.stats import beta
from scipy.stats import norm
def binomial_hpdr(n, N, pct=0.683, a=1, b=1, n_pbins=1000):
   # https://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
   """
   Function computes the posterior mode along with the upper and lower bounds of the
   **Highest Posterior Density Region**.
   
   Parameters
   ----------
   n: number of successes 
   N: sample size 
   pct: the size of the confidence interval (between 0 and 1)
   a: the alpha hyper-parameter for the Beta distribution used as a prior (Default=1)
   b: the beta hyper-parameter for the Beta distribution used as a prior (Default=1)
   n_pbins: the number of bins to segment the p_range into (Default=1e3)
   
   Returns
   -------
   A tuple that contains the mode as well as the lower and upper bounds of the interval
   (mode, lower, upper)
   
   """
   # fixed random variable object for posterior Beta distribution
   rv = beta(n+a, N-n+b)
   # determine the mode and standard deviation of the posterior
   stdev = rv.stats('v')**0.5
   mode = (n+a-1.)/(N+a+b-2.)
   # compute the number of sigma that corresponds to this confidence
   # this is used to set the rough range of possible success probabilities
   n_sigma = numpy.ceil(norm.ppf( (1+pct)/2. ))+1
    # set the min and max values for success probability 
   max_p = mode + n_sigma * stdev
   if max_p > 1:
      max_p = 1.
   min_p = mode - n_sigma * stdev
   if min_p > 1:
      min_p = 1.
   # make the range of success probabilities
   p_range = numpy.linspace(min_p, max_p, n_pbins+1)
   # construct the probability mass function over the given range
   if mode > 0.5:
      sf = rv.sf(p_range)
      pmf = sf[:-1] - sf[1:]
   else:
      cdf = rv.cdf(p_range)
      pmf = cdf[1:] - cdf[:-1]
   # find the upper and lower bounds of the interval 
   sorted_idxs = numpy.argsort( pmf )[::-1]
   cumsum = numpy.cumsum( numpy.sort(pmf)[::-1] )
   j = numpy.argmin( numpy.abs(cumsum - pct) )
   upper = p_range[ (sorted_idxs[:j+1]).max()+1 ]
   lower = p_range[ (sorted_idxs[:j+1]).min() ]    
   return (lower, upper)

