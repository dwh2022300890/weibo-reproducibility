"""Unchanged spline basis and design functions from the verified April analysis."""
import numpy as np
from scipy.interpolate import BSpline

def logheat(x):
    return np.log1p(x)

def limits(k):
    return (0, 1) if k == -1 else (2**k, 2**(k+1))

class NaturalSpline:
    """Natural cubic B-spline with the value at the lower boundary anchored at zero."""
    def __init__(self, knots):
        self.knots = np.asarray(knots, dtype=float)
        self.scale = self.knots[-1]-self.knots[0]
        nodes = (self.knots-self.knots[0])/self.scale
        t = np.r_[[0.]*4,nodes[1:-1],[1.]*4]
        raw = BSpline(t,np.eye(len(t)-4),3)
        constraints = np.stack((raw(0.,nu=2),raw(1.,nu=2),raw(0.)))
        constraints /= np.linalg.norm(constraints,axis=1)[:,None]
        _,singular,v = np.linalg.svd(constraints,full_matrices=True)
        assert np.min(singular)>1e-10
        self.spline = BSpline(t,v[3:].T.copy(),3,extrapolate=False)
        self.size = v.shape[0]-3
        self.left_derivative = self.spline(0.,nu=1)
        self.right_derivative = self.spline(1.,nu=1)
        assert self.size==len(knots)-1
        self.lookup = None
        self.unit = None

    def cache(self, count, unit):
        self.unit = unit
        self.lookup = self.basis(np.log1p(np.arange(count)/unit))

    def from_counts(self, values, unit=1):
        if self.lookup is None:
            return self.basis(np.log1p(values/unit))
        assert unit == self.unit
        high = values>=len(self.lookup)
        if not np.any(high):
            return self.lookup[values]
        out = self.lookup[np.minimum(values,len(self.lookup)-1)]
        out[high] = self.basis(np.log1p(values[high]/unit))
        return out

    def basis(self, values):
        x = (values-self.knots[0])/self.scale
        v = self.spline(np.clip(x,0,1))
        left,right = x<0,x>1
        v[left] += x[left,None]*self.left_derivative
        v[right] += (x[right,None]-1)*self.right_derivative
        return v

def model(cap):
    hi = limits(cap)[1]
    hnodes = [0] + [2**k for k in (0, 3, 6, 9, 12, 15, 18, 21) if 2**k<hi] + [hi]
    delay = [0, 5/60, 1, 24, 720]
    age = [0, .5, 3, 12, 24, 48]
    return NaturalSpline(logheat(hnodes)), NaturalSpline(np.log1p(delay)), NaturalSpline(np.log1p(age))

def design(splines, h, delay, age):
    heat,d,a = splines
    X = np.empty((len(h),1+heat.size+d.size+a.size))
    X[:,0] = 1
    X[:,1:1+heat.size] = heat.from_counts(h)
    X[:,1+heat.size:1+heat.size+d.size] = d.from_counts(delay,3600)
    X[:,1+heat.size+d.size:] = a.from_counts(age,3600)
    return X
