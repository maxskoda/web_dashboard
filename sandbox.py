# import plotly.express as px
# from mantid.simpleapi import *
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# from matplotlib.colors import LogNorm
import numpy as np
import json
from requests_html import HTMLSession, AsyncHTMLSession
import requests
from main import get_json_values

url_inst = 'http://ndxinter:4812/group?name=INST&format=json'
url_blocks = 'http://ndxinter:4813/group?name=BLOCKS&format=json'

r_inst = requests.get(url_inst)
r_blocks = requests.get(url_blocks)

values = get_json_values() #get_values(session)

print(values)

# print(json.dumps(r.json(), indent=4, sort_keys=True))
# print(r.json()['Channels'])

# for ch in r_blocks.json()['Channels']:
#     print(ch['Channel'].split(':SB:')[1], type(ch['Current Value']['Value']))

for ch in r_inst.json()['Channels']:
    try:
        print(ch['Channel'].split(':DAE:')[1], ch['Current Value']['Value'])
    except IndexError:
        pass
