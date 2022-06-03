import datetime
import time
import dash
from dash import dcc, html
import plotly
from dash.dependencies import Input, Output

from requests_html import HTMLSession
import re
from main import get_values



# from flask import Flask, jsonify
import asyncio
from threading import Thread

from pyorbital.orbital import Orbital
satellite = Orbital('TERRA')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

url='http://dataweb.isis.rl.ac.uk/IbexDataweb/default.html?Instrument=inter'
session = HTMLSession()

# tt=0


def get_stuff():
    # global tt
    r = session.get(url)
    r.html.render()

    # #take the rendered html and find the element that we are interested in
    inst_pvs = r.html.find('#inst_pvs', first=True).text
    main = r.html.find('#groups', first=True).text
    inst_pvs = r.html.find('#inst_pvs', first=True).text
    m = re.search('(?<=Inst. Time:).*(?=:)', inst_pvs)
    # lang_bar = r[0].html.find('#LangBar', first=True)
    tt = m.group(0)[-2:]
    return tt


get_stuff()
# values = get_values(session)
values = get_values(session, 'Run Number:', ':')
print(values)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('TERRA Satellite Live Feed'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=3*1000, # in milliseconds
            n_intervals=0
        )
    ])
)


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    # tt = get_stuff()
    # print(tt)
    values = get_values(session)
    print(values['S1VG'])

    lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Longitude: {0:.2f}'.format(lon), style=style),
        html.Span('Latitude: {0:.2f}'.format(lat), style=style),
        html.Span('Altitude: {0:0.2f}'.format(alt), style=style)
    ]


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    satellite = Orbital('TERRA')
    data = {
        'time': [],
        'Latitude': [],
        'Longitude': [],
        'Altitude': []
    }

    # Collect some data
    for i in range(180):
        time = datetime.datetime.now() - datetime.timedelta(seconds=i*20)
        lon, lat, alt = satellite.get_lonlatalt(
            time
        )
        data['Longitude'].append(lon)
        data['Latitude'].append(lat)
        data['Altitude'].append(alt)
        data['time'].append(time)

    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    fig.append_trace({
        'x': data['time'],
        'y': data['Altitude'],
        'name': 'Altitude',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': data['Longitude'],
        'y': data['Latitude'],
        'text': data['time'],
        'name': 'Longitude vs Latitude',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 2, 1)

    return fig


if __name__ == '__main__':

    app.run_server(debug=True)
