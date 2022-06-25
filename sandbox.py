import plotly.express as px
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import numpy as np

wksp = mtd['INTER00066223']