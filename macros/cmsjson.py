import json
import numpy as np

class CMSJson(object):
    def __init__(self, fname):
        self.jmap = {int(i) : j for i,j in json.load(open('fill6371_JSON.txt')).iteritems()}
    
    def __contains__(self, key):
        run, lumi = key
        if run not in self.jmap: return False
        for lrange in self.jmap[run]:
            if lrange[0] <= lumi <= lrange[1]: return True
        return False
    
    def contains(self, run, lumi):
        cnt = lambda r, l: ((r, l) in self) #workaround to avoid self being picked-up by vectorize
        cnt = np.vectorize(cnt)
        return cnt(run, lumi)
    
    def __repr__(self):
        return 'CMSJson(%s)' % self.jmap
