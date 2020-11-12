"""
Show that there is (not) choice information on the signal axis 
"""
from settings import DIR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib as mpl
import pandas as pd
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

choice = pd.read_pickle(DIR + 'results/res_choice_decoder.pickle')
stimulus = pd.read_pickle(DIR + 'results/res_stimulus_decoder.pickle')

