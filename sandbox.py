# import plotly.express as px
# from mantid.simpleapi import *
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# from matplotlib.colors import LogNorm
import plotly.graph_objects as go
import numpy as np
import json
from requests_html import HTMLSession, AsyncHTMLSession
import dash_bootstrap_components as dbc
import requests
from main import get_json_values

url_inst = 'http://ndxinter:4812/group?name=INST&format=json'
url_blocks = 'http://ndxinter:4813/group?name=BLOCKS&format=json'

r_inst = requests.get(url_inst)
r_blocks = requests.get(url_blocks)

values = get_json_values() #get_values(session)

print(values)


np.random.seed(1)

N = 100
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
sz = np.random.rand(N) * 30

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="markers",
    marker=go.scatter.Marker(
        size=sz,
        color=colors,
        opacity=0.6,
        colorscale="Viridis"
    )
))

# fig.show()
fig.write_image("fig1.png")

# print(json.dumps(r.json(), indent=4, sort_keys=True))
# print(r.json()['Channels'])

# for ch in r_blocks.json()['Channels']:
#     print(ch['Channel'].split(':SB:')[1], type(ch['Current Value']['Value']))

for ch in r_inst.json()['Channels']:
    try:
        print(ch['Channel'].split(':DAE:')[1], ch['Current Value']['Value'])
    except IndexError:
        pass
