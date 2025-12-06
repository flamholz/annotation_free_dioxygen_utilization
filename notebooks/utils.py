import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from aerobot.utils import FEATURE_TYPES, FEATURE_ORDERS
from aerobot.plot import *
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticks
from scipy.stats import linregress
from Bio import SeqIO

figures_dir = '/home/prichter/Documents/aerobot-paper/figures/'
results_dir = '/home/prichter/Documents/aerobot-paper/results/'
data_dir = '/home/prichter/Documents/aerobot-paper/data/'
contigs_dir = os.path.join(data_dir, 'contigs')
rna16s_dir = os.path.join(data_dir, 'rna16s')