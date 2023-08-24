import pandas as pd
import numpy as np
import os



def load_data(path):
    data = pd.read_csv(os.path.join('data',path))
    return data
