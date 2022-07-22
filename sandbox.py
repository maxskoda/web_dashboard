# import plotly.express as px
# from mantid.simpleapi import *
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# from matplotlib.colors import LogNorm
import numpy as np
import json

# Opening JSON file
with open('C:/Users/ktd43279/Downloads/MannidTable_1.json') as f:

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    for r in data['experimentView']['perAngleDefaults']['rows']:
        print(r)
