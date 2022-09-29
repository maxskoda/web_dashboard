# import mantid algorithms, numpy and matplotlib
import base64
import json

from dash.exceptions import PreventUpdate
from mantid.simpleapi import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from CaChannel import CaChannel, ca  # CaChannelException, ca

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash
from dash import html, callback_context
from dash import dcc
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import datetime
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.subplots import make_subplots
from plotly.express import data

from requests_html import HTMLSession
from main import get_values, get_json_values
import threading

global content_dict

# some settings
instrument = 'INTER'
cycle = '22_3'

now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y   %H:%M")

values = get_json_values(instrument) #get_values(session)
try:
    wksp = str(int(values['RUNNUMBER.VAL']) - 1)
except ValueError:
    wksp = "66223"


# t1 = threading.Thread(target=f, args=[xd, yd, ed])


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.CYBORG]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

degree_sign = u'\N{DEGREE SIGN}'


def create_card(title, content, color):
    card_content = content.split("|")

    cb = []
    for c in card_content:
        cb.append(html.H4(c, className="card-text"))
        cb.append(html.Br())
    card = dbc.Card(
        [
            dbc.CardHeader(title),
            dbc.CardBody(cb),
        ],
        color=color, inverse=False,
    )
    return card


# Trans number input:
trans1_input = dbc.Row(
    [
        # dbc.Col(dbc.Label("Trans 1", className="ml-2", style={'font-size': '15px'})),
        dbc.Col(dbc.Input(id="input1", placeholder="Trans1", type="number",
                          debounce=True, style={'font-size': '15px'})),
        html.Br(),
        # html.P(id="output1"),
    ]
)

trans2_input = dbc.Row(
    [
        # dbc.Col(dbc.Label("Trans 2",  style={'font-size': '15px'})),
        dbc.Col(dbc.Input(id="input2", placeholder="Trans2", type="number",
                          debounce=True, style={'font-size': '15px'})),
        html.Br(),
        # html.P(id="output2"),
    ]
)


@app.callback(Output("output2", "children"), [Input("input2", "value")])
def output_trans2(value):
    return value


def create_table(title, rows):
    table_head = [
        html.Thead(html.Tr(html.Th(html.H4(html.B(title)), colSpan=str(len(rows[0])))))
    ]

    table_rows = []
    for row in rows:
        table_rows.append(html.Tr([html.Td(html.H5(text)) for text in row]))

    table_body = [html.Tbody(table_rows)]

    sample_table = dbc.Table(table_head + table_body,
                             borderless=True,
                             dark=True,
                             hover=True,
                             responsive=True,
                             striped=True,
                             style={'padding-right': '10px', 'padding-left': '100px'},
                             color="light",
                             )
    return sample_table


# Empty card
empty_card = create_card("", "", "secondary")


@app.callback(Output('row0', 'children'),
              Input('interval-component', 'n_intervals'))
def row0(n):
    valuesl = get_json_values(instrument) #get_values(session)
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y   %H:%M")
    # ### Top row #####
    card3 = create_card("Date and Time", dt_string, "secondary")
    card2 = create_card("Title", valuesl['TITLE.VAL'], "primary")
    rno = str(int(valuesl['RUNNUMBER.VAL']))
    run_status = valuesl['RUNSTATE.VAL']

    col_dict = {'RUNNING': "success", 'SETUP': "primary", 'PAUSED': "danger"}
    rs = run_status.replace('\\xa0', '').strip()

    card1 = create_card("Instrument - Run No", "INTER - " + rno, col_dict[rs])

    # Run number:
    run_table = create_table("Instrument - Run No", [["INTER - " + valuesl["RUNNUMBER.VAL"]]])

    # Title:
    title_table = create_table("Title", [[valuesl["TITLE.VAL"]]])

    # Time/Date:
    time_table = create_table("Date and Time", [[dt_string]])

    graphRow0 = dbc.Row([dbc.Col(id='card1', children=[run_table], md=2),
                         dbc.Col(id='card2', children=[title_table], md=8),
                         dbc.Col(id='card3', children=[time_table], md=2)], style={'padding': 10})
    return graphRow0


@app.callback(Output('row1', 'children'),
              Input('interval-component', 'n_intervals'))
def row1(n):
    valuesl = get_json_values(instrument) #get_values(session)

    # SLIT 1:
    slit1_table = create_table("Slit 1", [["S1VG", valuesl["S1VG"]],
                                          ["S1HG", valuesl["S1HG"]]])
    # SLIT 2:
    slit2_table = create_table("Slit 2", [["S2VG", valuesl["S2VG"]],
                                          ["S2HG", valuesl["S2HG"]]])
    # Supermirrors
    SM_table = create_table("Supermirrors", [["SM1 Ang.: ", valuesl['SM1ANGLE'] + degree_sign],
                                             ["SM2 Ang.: ", valuesl['SM2ANGLE'] + degree_sign]])
    # Sample stack
    sample_table = create_table("Sample Stack", [['PHI:', valuesl['PHI'] + degree_sign, 'HEIGHT:', valuesl['HEIGHT']],
                                                 ['TRANS:', valuesl['TRANS'], 'HEIGHT2:', valuesl['HEIGHT2']]])
    # Detector
    detector_table = create_table("Detector", [["THETA: ", valuesl['THETA'] + degree_sign]])

    graphRow1 = dbc.Row([dbc.Col(id='slit1_table', children=[slit1_table], md=2),
                         dbc.Col(id='SM_table', children=[SM_table], md=2),
                         dbc.Col(id='slit2_table', children=[slit2_table], md=2),
                         dbc.Col(id='sample_table', children=[sample_table], md=4),
                         dbc.Col(id='detector_table', children=[detector_table], md=2), ],
                        style={'padding': 10})
    return graphRow1


# Create global figures
xd, yd, ed = np.loadtxt('text.csv', delimiter=' ', usecols=(0, 1, 2), unpack=True)
fig = go.Figure(data=[go.Scatter(x=xd, y=yd, error_y=dict(
    type='data',
    array=ed,
    visible=True))])

# Detector image
wksp = LoadISISNexus('INTER00066223')
z = wksp.extractY()
plotly_fig = px.imshow(np.log(z), aspect='auto', origin='lower', color_continuous_scale='rainbow')

perangle = [['0.8', '', '66475', '66476', '70-90', '', '0.08', '', '', '70-90', '20-60'],
            ['2.3', '', '66746', '66747', '70-90', '', '', '', '', '70-95', '140-150']]


def settings_table(settings):
    table_header = [
        html.Thead(html.Tr([html.Th("Angle"), html.Th("Title"), html.Th("1st Trans run(s)"), html.Th("2nd Trans run(s)"),
                            html.Th("Trans spectra"), html.Th("Q min"), html.Th("Q max"), html.Th("dQ/Q"),
                            html.Th("Scale"), html.Th("ROI"), html.Th("Background")]))
    ]
    rows = []
    for r in settings:
        row = []
        for col in r:
            row.append(html.Td(col))
        rows.append(html.Tr(row))

    table_body = [html.Tbody(rows)]

    table = dbc.Table(
        table_header + table_body,
        bordered=True,
        size='lg',
        style={
            'fontSize': '16px',
        },
    )
    return table


def reduce(run):#, trans1=None, trans2=None):
    global content_dict
    settings_dict = {}

    if content_dict is not None:
        settings_dict['AnalysisMode'] = 'MultiDetectorAnalysis'
        settings_dict['WavelengthMin'] = content_dict['instrumentView']['lamMinEdit']
        settings_dict['WavelengthMax'] = content_dict['instrumentView']['lamMaxEdit']
        settings_dict['I0MonitorIndex'] = content_dict['instrumentView']['I0MonitorIndex']
        settings_dict['MonitorBackgroundWavelengthMin'] = content_dict['instrumentView']['monBgMinEdit']
        settings_dict['MonitorBackgroundWavelengthMax'] = content_dict['instrumentView']['monBgMaxEdit']
        settings_dict['MonitorIntegrationWavelengthMin'] = content_dict['instrumentView']['monIntMinEdit']
        settings_dict['MonitorIntegrationWavelengthMax'] = content_dict['instrumentView']['monIntMaxEdit']

        settings_dict['StartOverlap'] = content_dict['experimentView']['startOverlapEdit']
        settings_dict['EndOverlap'] = content_dict['experimentView']['endOverlapEdit']
        settings_dict['ScaleRHSWorkspace'] = content_dict['experimentView']['transScaleRHSCheckBox']

        # need to check which row
        s_row=0
        settings_dict['FirstTransmissionRunList'] = content_dict['experimentView']['perAngleDefaults']['rows'][s_row][2]
        settings_dict['SecondTransmissionRunList'] = content_dict['experimentView']['perAngleDefaults']['rows'][s_row][3]
        settings_dict['TransmissionProcessingInstructions'] = content_dict['experimentView']['perAngleDefaults']['rows'][s_row][4]
        settings_dict['ProcessingInstructions'] = content_dict['experimentView']['perAngleDefaults']['rows'][s_row][9]

    print(settings_dict)

    ReflectometryISISLoadAndProcess(InputRunList=str(run), **settings_dict, OutputWorkspaceBinned='IvsQ_binned_' + str(run))


@app.callback(
    Output('graph_row', 'children'),
    # [Input('input1', 'value'), Input('input2', 'value')],
    Input('runlist-dropdown', 'value'),
    Input('interval-component', 'n_intervals'),
)
def graph_row(value, n_int): #trans1_value, trans2_value,
    global fig, content_dict
    # print("TRANSMISSIONS: ", trans1_value, trans2_value)
    print("Runs: ", value)
    # Get live data from file
    xd, yd, ed = np.loadtxt('text.csv', delimiter=' ', usecols=(0, 1, 2), unpack=True)
    # 1. delete all traces
    fig.data = []

    # 2. Add live data to plot
    live_data = [go.Scatter(x=xd, y=yd, error_y=dict(
        type='data',
        array=ed,
        visible=True))]
    # fig_data.append(live_data[0])
    fig.add_traces(live_data[0])

    # 3. Reduce and plot all other selected runs
    try:
        for run in value:
            if ('IvsQ_binned_' + str(run)) not in mtd.getObjectNames() and 'content_dict' in globals():
                reduce(run)#, trans1_value, trans2_value)
            else:
                print("Please load a valid settings file!")

            xd = mtd['IvsQ_binned_' + str(run)].dataX(0)
            yd = mtd['IvsQ_binned_' + str(run)].dataY(0)
            ed = mtd['IvsQ_binned_' + str(run)].dataE(0)

            fig.add_trace(go.Scatter(x=xd, y=yd, error_y=dict(
                type='data',
                array=ed,
                visible=True), name=str(run)))
    except TypeError:
        print('No runs selected.')

    fig.update_yaxes(type="log", tickfont_size=18)
    fig.update_xaxes(type="log", tickfont_size=18)

    fig.update_layout(
        yaxis_tickformat='.0e',
        uirevision="Don't change",
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
        hovermode='closest',
    )
    fig.update_yaxes(nticks=6)
    fig.update_yaxes(nticks=6)

    gr = dcc.Graph(id='reflectivity-graph', figure=fig, style={'height': '60vh'})
    return gr


@app.callback(Output('graph_row_2', 'children'),
              Input('runlist-dropdown-2', 'value'))
# Input('tabs-graph', 'value'))
def graph_row_2(run):
    try:
        wksp = mtd[str(run)]
        z = wksp.extractY()
    except KeyError:
        wksp = Load(str(run), OutputWorkspace=str(run))
        z = wksp.extractY()

    plotly_fig_2 = px.imshow(np.log(z), aspect='auto', origin='lower', color_continuous_scale='rainbow')

    gr = dcc.Graph(id='detector-image-graph', figure=plotly_fig_2, style={'width': '80vh', 'height': '60vh'})
    return gr


app.layout = html.Div([
    html.Div(id='row0'),
    html.Div(id='row1'),
    dbc.Tabs(id="tabs-graph", children=[  #value='tab-1-graph'
        dcc.Tab(label='Reduced data', children=[
            dbc.Row([#dbc.Col(id="trans_col", children=[dbc.Row(trans1_input), dbc.Row(trans2_input)],
                      #       md=1, width={"offset": 1}),
                     dbc.Col(html.Div(id='graph_row'), md=8, width={"offset": 2}),
                     dbc.Col(html.Div(dcc.Dropdown(id='runlist-dropdown', placeholder="Select runs - hover for title",
                                                   multi=True, style={'font-size': '15px'}),
                                      className="dash-bootstrap"), md=2),
                     ], style={'padding': 10})
        ]),
        # style={'font-size': '15px', 'line-height': '5vh', 'padding': '10'}),
                # selected_style={'padding': '0', 'line-height': '5vh'}),
        dcc.Tab(label='Detector image', id='tab-2-graph', children=[ #value='tab-2-graph',
            dbc.Row([
                dbc.Col(html.Div(id='graph_row_2'), md=7, width={"offset": 2}),
                dbc.Col(html.Div(dcc.Dropdown(id='runlist-dropdown-2', placeholder="Select runs - hover for title",
                                              multi=False, style={'font-size': '15px'}),
                                 className="dash-bootstrap"), md=2),
            ]),
        ]),
        #style={'font-size': '15px', 'line-height': '5vh', 'padding': '0'}),
                # selected_style={'padding': '0', 'line-height': '5vh'}),
        dcc.Tab(label='Settings', children=[
            dcc.Upload(id="upload-data", children=html.Div([
                html.H4('Drag and Drop or '),
                html.A(html.H4('Select a File'))
            ]),  style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
            }),
            html.Div(
                [
                    dbc.Row(dbc.Col(html.Div(id="output-data-upload"), width={"size": 8, "offset": 2})),

                ])
        ],)
                # style={'font-size': '15px', 'line-height': '5vh', 'padding': '0'}),
                # selected_style={'padding': '0', 'line-height': '5vh'}),
    ],
             style={
                 # 'width': '50%',
                 'font-size': '120%',
                 'height': '5vh'
             }),
    html.Div(id='hidden-div', style={'display': 'none'}),

    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Interval(
        id='interval-component-2',
        interval=60 * 1000,  # in milliseconds
        n_intervals=0
    )
],
)


@app.callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def on_data_upload(contents, filename):
    global content_dict
    # print("DONE")
    # if contents is None:
    #     raise PreventUpdate

    if not filename.endswith(".json"):
        return "Please upload a file with the .json extension"

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    # this is a dict!
    content_dict = json.loads(decoded)
    for r in content_dict['experimentView']['perAngleDefaults']['rows']:
        print(r)

    return settings_table(content_dict['experimentView']['perAngleDefaults']['rows'])  #str(data['experimentView']['perAngleDefaults']['rows'][0])


def get_runs(cycle):
    try:
        ISISJournalGetExperimentRuns(Cycle=cycle, InvestigationId=str(values['_RBNUMBER.VAL'].strip()),
                                 OutputWorkspace='RB' + str(values['_RBNUMBER.VAL'].strip()))
        runs = mtd['RB' + str(values['_RBNUMBER.VAL'].strip())]
    except ValueError:
        rbno = "2210267" #input("Please enter a valid RB number:")
        ISISJournalGetExperimentRuns(Cycle=cycle, InvestigationId=str(rbno),
                                     OutputWorkspace='RB' + str(rbno))
        runs = mtd['RB' + str(rbno)]

    opt = [item for item in runs.column(1)]
    titles = [item for item in runs.column(2)]
    print(opt)
    return opt, titles


opt, titles = get_runs(cycle)


@app.callback(
    [Output("runlist-dropdown", "options"), Output("runlist-dropdown", "value")],
    [Input("interval-component-2", "n_intervals")],
    [State("runlist-dropdown", "value")]
)
def make_dropdown_options(n, value):
    opt, titles = get_runs(cycle)
    a_zip = zip(opt, titles)
    zipped_list = list(a_zip)

    options = [{"label": v, "value": v, "title": t} for v, t in zipped_list]
    # options = [{"label": v, "value": v, "title": t} for v in opt for t in titles]

    return options, value


@app.callback(
    [Output("runlist-dropdown-2", "options"), Output("runlist-dropdown-2", "value")],
    [Input("interval-component-2", "n_intervals")],
    [State("runlist-dropdown-2", "value")]
)
def make_dropdown_2_options(n, value):
    opt, titles = get_runs(cycle)
    a_zip = zip(opt, titles)
    zipped_list = list(a_zip)

    options = [{"label": v, "value": v, "title": t} for v, t in zipped_list]

    return options, value


if __name__ == '__main__':
    # t1.start()
    # print(xd)
    app.run_server(host="0.0.0.0", port="8050")
