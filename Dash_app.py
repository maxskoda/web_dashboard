# import mantid algorithms, numpy and matplotlib
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
from main import get_values
import threading

now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y   %H:%M")

url = 'http://dataweb.isis.rl.ac.uk/IbexDataweb/default.html?Instrument=inter'
session = HTMLSession()

values = get_values(session)
try:
    wksp = str(int(values['run_number']) - 1)
except ValueError:
    wksp = "66223"

# chan = CaChannel('IN:"""+inst+""":REFL_01:PARAM:THETA')
# chan.setTimeout(5.0)
# chan.searchw()
# as_string=True # set to False if you want a numeric value\n
# theta = float(chan.getw(ca.DBR_STRING if as_string else None))
# theta_in=abs(theta)
# print("THETA: ", theta_in)

##############
inst = "INTER"
lambda_min = 1.8
lambda_max = 15
trans_SM = 'TRANS_SM'
trans = 'TRANS'
dq_q = 0.03
TRANS_ROI = '70-90'
ROI = '70-90'

# Load(Filename=r'\\isis.cclrc.ac.uk\inst$\ndxinter\instrument\data\cycle_21_2\INTER00065274.raw', OutputWorkspace='INTER00065274')
# Load(Filename=r'\\isis.cclrc.ac.uk\inst$\ndxinter\instrument\data\cycle_21_2\INTER00065275.raw', OutputWorkspace='INTER00065275')
# CreateTransmissionWorkspaceAuto(FirstTransmissionRun='INTER00065274', SecondTransmissionRun='INTER00065275', AnalysisMode='MultiDetectorAnalysis', ProcessingInstructions='70-90', ScaleRHSWorkspace=False, OutputWorkspace='TRANS_low')


ReflectometryISISLoadAndProcess(InputRunList='65272', ThetaIn=0.8,
                                AnalysisMode='MultiDetectorAnalysis', ProcessingInstructions='70-90',
                                WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2,
                                MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18,
                                MonitorIntegrationWavelengthMin=4,
                                MonitorIntegrationWavelengthMax=10,
                                FirstTransmissionRunList='65274', SecondTransmissionRunList='65275',
                                StartOverlap=10, EndOverlap=12, ScaleRHSWorkspace=False,
                                TransmissionProcessingInstructions='70-90',
                                MomentumTransferMin=0.010321317306126728,
                                MomentumTransferStep=0.055433662337842131,
                                MomentumTransferMax=0.1168874036214391,
                                OutputWorkspaceBinned='IvsQ_binned_65272',
                                OutputWorkspace='IvsQ_65272',
                                OutputWorkspaceTransmission='TRANS_SM')

ReflectometryISISLoadAndProcess(InputRunList='65273', ThetaIn=2.3, AnalysisMode='MultiDetectorAnalysis',
                                ProcessingInstructions='67-95', WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2,
                                MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18,
                                MonitorIntegrationWavelengthMin=4, MonitorIntegrationWavelengthMax=10,
                                FirstTransmissionRunList='65276', SecondTransmissionRunList='65277', StartOverlap=10,
                                EndOverlap=12, ScaleRHSWorkspace=False, TransmissionProcessingInstructions='70-90',
                                MomentumTransferMin=0.029666234509808882, MomentumTransferStep=0.055446760622640492,
                                MomentumTransferMax=0.33612056568876092, OutputWorkspaceBinned='IvsQ_binned_65273',
                                OutputWorkspace='IvsQ_65273', OutputWorkspaceTransmission='TRANS')

# Stitch1DMany(InputWorkspaces='IvsQ_65272,IvsQ_65273', OutputWorkspace='IvsQ_65272_65273', Params='-0.055434', OutScaleFactors='0.841361')


# script = """from CaChannel import CaChannel, CaChannelException, ca\n"""

script = """# chan = CaChannel('IN:""" + inst + """:REFL_01:PARAM:THETA')\n
#chan.setTimeout(5.0)\n
#chan.searchw()\n
#as_string=True # set to False if you want a numeric value\n
#theta = float(chan.getw(ca.DBR_STRING if as_string else None))
#theta_in=abs(theta)\n
theta_in=float(values['THETA'])\n
qmin=4*3.1415/""" + str(lambda_max) + """*(theta_in*3.141/180)
qmax=4*3.1415/""" + str(lambda_min) + """*(theta_in*3.141/180)

if theta_in<1.0:\n
\t trans='""" + trans_SM + """'\n
else:\n
\t trans='""" + trans + """'\n

ReflectometryISISLoadAndProcess(InputRunList=input, ThetaIn=theta_in, SummationType='SumInQ', ReductionType='DivergentBeam', 
                        AnalysisMode='MultiDetectorAnalysis', 
                        ProcessingInstructions='""" + ROI + """', 
                        WavelengthMin=1.5, WavelengthMax=17, 
                        I0MonitorIndex=2, MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18, 
                        MonitorIntegrationWavelengthMin=4, MonitorIntegrationWavelengthMax=10, SubtractBackground=True, 
                        BackgroundCalculationMethod='AveragePixelFit', 
                        FirstTransmissionRunList=trans, StartOverlap=10, EndOverlap=12, ScaleRHSWorkspace=False, 
                        TransmissionProcessingInstructions='""" + TRANS_ROI + """', Debug=True, 
                        MomentumTransferMin=qmin, MomentumTransferStep=""" + str(dq_q) + """, 
                        MomentumTransferMax=qmax, OutputWorkspaceBinned='0_IvsQ_binned', 
                        OutputWorkspace='0_IvsQ', OutputWorkspaceWavelength='0_IvsLam', 
                        OutputWorkspaceTransmission='TRANS_LAM_0', 
                        OutputWorkspaceFirstTransmission='TRANS_LAM_0a', OutputWorkspaceSecondTransmission='TRANS_LAM_0b')

output="0_IvsQ_binned" """


def f(script):
    StartLiveData(Instrument=inst, ProcessingScript=script, AccumulationMethod='Replace',
                  UpdateEvery=10, OutputWorkspace='0_IvsQ_binned')
    xd = mtd['0_IvsQ_binned'].dataX(0)
    yd = mtd['0_IvsQ_binned'].dataY(0)
    ed = mtd['0_IvsQ_binned'].dataE(0)
    return xd, yd, ed


##############

# t1 = threading.Thread(target=f, args=[xd, yd, ed])
# xd, yd, ed = f(script=script)


# xd, yd, ed = np.loadtxt('text.csv', delimiter=' ', usecols=(0, 1, 2), unpack=True)


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
    valuesl = get_values(session)
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y   %H:%M")
    # ### Top row #####
    card3 = create_card("Date and Time", dt_string, "secondary")
    card2 = create_card("Title", valuesl['title'], "primary")
    rno = str(int(valuesl['run_number']))
    run_status = valuesl['run_status'].replace('\\xa0', '')

    col_dict = {'RUNNING': "success", 'SETUP': "primary", 'PAUSED': "danger"}
    rs = run_status.replace('\\xa0', '').strip()

    card1 = create_card("Instrument - Run No", "INTER - " + rno, col_dict[rs])

    graphRow0 = dbc.Row([dbc.Col(id='card1', children=[card1], md=2),
                         dbc.Col(id='card2', children=[card2], md=8),
                         dbc.Col(id='card3', children=[card3], md=2)], style={'padding': 10})
    return graphRow0


@app.callback(Output('row1', 'children'),
              Input('interval-component', 'n_intervals'))
def row1(n):
    valuesl = get_values(session)
    # print(valuesl["S1VG"])
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


# wksp = "66223"
# LoadISISNexus(wksp, OutputWorkspace='TOF_' + wksp)
# CropWorkspace(InputWorkspace='TOF_' + wksp, OutputWorkspace=wksp + '_lam', StartWorkspaceIndex=5)
# ConvertUnits(InputWorkspace=wksp + '_lam', OutputWorkspace=wksp + '_lam', Target='Wavelength', AlignBins=True)
# CropWorkspace(InputWorkspace=wksp + '_lam', OutputWorkspace=wksp + '_lam', XMax=17)
#
# plotly_fig, (current_ax, NR_ax) = plt.subplots(2, 1, subplot_kw={'projection': 'mantid'}, figsize=(7, 8))
# c = current_ax.pcolormesh(mtd[wksp+'_lam'])#,  norm=colors.LogNorm())


def reduce(run, trans1=None, trans2=None):
    if trans1 is not None and trans2 is not None:
        ReflectometryISISLoadAndProcess(InputRunList=str(run), AnalysisMode='MultiDetectorAnalysis',
                                        WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2,
                                        MonitorBackgroundWavelengthMin=17,
                                        MonitorBackgroundWavelengthMax=18,
                                        MonitorIntegrationWavelengthMin=4,
                                        MonitorIntegrationWavelengthMax=10,
                                        FirstTransmissionRunList=str(trans1),
                                        SecondTransmissionRunList=str(trans2),
                                        StartOverlap=10, EndOverlap=12,
                                        ScaleRHSWorkspace=False, TransmissionProcessingInstructions='70-90',
                                        ProcessingInstructions='70-90', OutputWorkspaceBinned='IvsQ_binned_' + str(run))
    elif trans1 is not None and trans2 is None:
        ReflectometryISISLoadAndProcess(InputRunList=str(run), AnalysisMode='MultiDetectorAnalysis',
                                        WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2,
                                        MonitorBackgroundWavelengthMin=17,
                                        MonitorBackgroundWavelengthMax=18,
                                        MonitorIntegrationWavelengthMin=4,
                                        MonitorIntegrationWavelengthMax=10,
                                        FirstTransmissionRunList=str(trans1),
                                        StartOverlap=10, EndOverlap=12,
                                        ScaleRHSWorkspace=False, TransmissionProcessingInstructions='70-90',
                                        ProcessingInstructions='70-90', OutputWorkspaceBinned='IvsQ_binned_' + str(run))
    else:
        ReflectometryISISLoadAndProcess(InputRunList=str(run), AnalysisMode='MultiDetectorAnalysis',
                                        WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2,
                                        MonitorBackgroundWavelengthMin=17,
                                        MonitorBackgroundWavelengthMax=18,
                                        MonitorIntegrationWavelengthMin=4,
                                        MonitorIntegrationWavelengthMax=10,
                                        ProcessingInstructions='70-90', OutputWorkspaceBinned='IvsQ_binned_' + str(run))


@app.callback(
    Output('graph_row', 'children'),
    [Input('input1', 'value'), Input('input2', 'value')],
    Input('runlist-dropdown', 'value'),
    Input('interval-component', 'n_intervals'),
)
def graph_row(trans1_value, trans2_value, value, n_int):
    global fig
    print("TRANSMISSIONS: ", trans1_value, trans2_value)
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
            if ('IvsQ_binned_' + str(run)) not in mtd.getObjectNames():
                reduce(run, trans1_value, trans2_value)

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
    )
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


# @app.callback(
#     Output("interval-component", "disabled"),
#     [Input("reduce-button", "n_clicks")],
#     [State("interval-component", "disabled")],
# )
# def toggle_interval(n, disabled):
#     if n:
#         return not disabled
#     return disabled


@app.callback(
    Output('interval-component', 'disabled'),
    Input('reduce-button', 'n_clicks'),
    State('runlist-dropdown', 'options'),
    [State('input1', 'value'), State('input2', 'value')]
)
def reduce_all(n_clicks, opts, trans1_value, trans2_value):
    # Reduce all  runs
    try:
        for run in opts[0]:
            reduce(run['value'], trans1_value, trans2_value)
        print(run['value'], " reduced!")
    except TypeError:
        print('No runs selected.')
    return False


app.layout = html.Div([
    html.Div(id='row0'),
    html.Div(id='row1'),
    dcc.Tabs(id="tabs-graph", value='tab-1-graph', children=[
        dcc.Tab(label='Reduced data', children=[
            dbc.Row([dbc.Col(id="trans_col", children=[dbc.Row(trans1_input), dbc.Row(trans2_input)],
                             md=1, width={"offset": 1}),
                     dbc.Col(html.Div(id='graph_row'), md=7, width={"offset": 0}),
                     dbc.Col(html.Div(dcc.Dropdown(id='runlist-dropdown', placeholder="Select runs",
                                                   multi=True, style={'font-size': '15px'}),
                                      className="dash-bootstrap"), md=2),
                     dbc.Col(dbc.Button("Reduce all", color="dark", className="me-1", id="reduce-button"), md=1),
                     ], style={'padding': 10})
        ], style={'font-size': '15px', 'line-height': '5vh', 'padding': '10'}, selected_style={'padding': '0','line-height': '5vh'}),
        dcc.Tab(label='Detector image', value='tab-2-graph', children=[
            dbc.Row([
                dbc.Col(html.Div(id='graph_row_2'), md=7, width={"offset": 1}),
                dbc.Col(html.Div(dcc.Dropdown(id='runlist-dropdown-2', placeholder="Select runs",
                                              multi=False, style={'font-size': '15px'}),
                                 className="dash-bootstrap"), md=2),
            ]),
        ], style={'font-size': '15px', 'line-height': '5vh', 'padding': '0'}, selected_style={'padding': '0', 'line-height': '5vh'})],
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


def get_runs(cycle):
    ISISJournalGetExperimentRuns(Cycle=cycle, InvestigationId=str(values['rbno'].strip()),
                                 OutputWorkspace='RB' + str(values['rbno'].strip()))
    runs = mtd['RB' + str(values['rbno'].strip())]
    opt = [item for item in runs.column(1)]
    return opt


opt = get_runs('22_2')


@app.callback(
    [Output("runlist-dropdown", "options"), Output("runlist-dropdown", "value")],
    [Input("interval-component-2", "n_intervals")],
    [State("runlist-dropdown", "value")]
)
def make_dropdown_options(n, value):
    opt = get_runs('22_2')
    options = [{"label": v, "value": v} for v in opt]

    # if value not in [o["value"] for o in options]:
    #     # if the value is not in the new options list, we choose a different value
    #     if options:
    #         value = options[0]["value"]
    #     else:
    #         value = None
    return options, value


@app.callback(
    [Output("runlist-dropdown-2", "options"), Output("runlist-dropdown-2", "value")],
    [Input("interval-component-2", "n_intervals")],
    [State("runlist-dropdown-2", "value")]
)
def make_dropdown_options(n, value):
    opt = get_runs('22_2')
    options = [{"label": v, "value": v} for v in opt]

    # if value not in [o["value"] for o in options]:
    #     # if the value is not in the new options list, we choose a different value
    #     if options:
    #         value = options[0]["value"]
    #     else:
    #         value = None
    return options, value


if __name__ == '__main__':
    # t1.start()
    # print(xd)
    app.run_server(host="0.0.0.0", port="8050")
