import numpy as np
import pandas as pd

def round_up(x, decimal_places):
    
    return np.ceil(x * 10**decimal_places) / 10**decimal_places