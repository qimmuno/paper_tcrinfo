import numpy as np
from scipy.odr import *
from numpy import percentile
from .misc import round_up

def linear_func(p, x):
     a, b = p
     return a*x + b
 
def fixed_negative_gradient_func(b, x):
     return -x + b

def fixed_positive_gradient_func(b, x):
     return x + b
 
def plot_fit(x, y, func, beta0, x_to_plot, ax, color, linestyle = '-', intervals = True, title = True, decimal_places = 1, *args, **kwargs):
    model = Model(func)
    
    data = RealData(x, y, *args, **kwargs)
    odr = ODR(data, model, beta0=beta0)
    out = odr.run()
    
    ax.plot(x_to_plot, func(out.beta,  x_to_plot), c=color, linestyle=linestyle)
    
    if intervals:
        ps = np.random.multivariate_normal(out.beta, out.cov_beta, 10000)
        ysample = np.asarray([func(pi, x_to_plot) for pi in ps])
        lower_bound = percentile(ysample, 2.5, axis=0)
        upper_bound = percentile(ysample, 97.5, axis=0)
        ax.fill_between(x_to_plot, lower_bound, upper_bound, color=color, alpha=0.3)  
        
    if title:
        ax.set_title(f" a = {out.beta[0]:.{decimal_places}f} $\pm$ {round_up(out.sd_beta[0], decimal_places)}")
        
    return out
 
