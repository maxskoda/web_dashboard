# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash
from dash import html
from dash import dcc
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express import data

from requests_html import HTMLSession
from main import get_values
import threading

now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y   %H:%M")

url='http://dataweb.isis.rl.ac.uk/IbexDataweb/default.html?Instrument=inter'
session = HTMLSession()

values = get_values(session)

wksp = str(int(values['run_number'])-1)

##############
inst = "INTER"
lambda_min = 1.8
lambda_max = 15
trans_SM = 'TRANS_SM'
trans = 'TRANS'
dq_q = 0.03
TRANS_ROI = '70-90'
ROI = '70-90'

ReflectometryISISLoadAndProcess(InputRunList='65272', ThetaIn=0.8,
                                AnalysisMode='MultiDetectorAnalysis', ProcessingInstructions='70-90',
                                WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2,
                                MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18, MonitorIntegrationWavelengthMin=4,
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

ReflectometryISISLoadAndProcess(InputRunList='65273', ThetaIn=2.3, AnalysisMode='MultiDetectorAnalysis', ProcessingInstructions='67-95', WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2, MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18, MonitorIntegrationWavelengthMin=4, MonitorIntegrationWavelengthMax=10, FirstTransmissionRunList='65276', SecondTransmissionRunList='65277', StartOverlap=10, EndOverlap=12, ScaleRHSWorkspace=False, TransmissionProcessingInstructions='70-90', MomentumTransferMin=0.029666234509808882, MomentumTransferStep=0.055446760622640492, MomentumTransferMax=0.33612056568876092, OutputWorkspaceBinned='IvsQ_binned_65273', OutputWorkspace='IvsQ_65273', OutputWorkspaceTransmission='TRANS')
# Stitch1DMany(InputWorkspaces='IvsQ_65272,IvsQ_65273', OutputWorkspace='IvsQ_65272_65273', Params='-0.055434', OutScaleFactors='0.841361')


# script = """from CaChannel import CaChannel, CaChannelException, ca\n"""

script = """# chan = CaChannel('IN:"""+inst+""":REFL_01:PARAM:THETA')\n
#chan.setTimeout(5.0)\n
#chan.searchw()\n
#as_string=True # set to False if you want a numeric value\n
#theta = float(chan.getw(ca.DBR_STRING if as_string else None))
#theta_in=abs(theta)\n
theta_in=float(values['THETA'])\n
qmin=4*3.1415/"""+str(lambda_max)+"""*(theta_in*3.141/180)
qmax=4*3.1415/"""+str(lambda_min)+"""*(theta_in*3.141/180)

if theta_in<1.0:\n
\t trans='"""+trans_SM+"""'\n
else:\n
\t trans='"""+trans+"""'\n

ReflectometryISISLoadAndProcess(InputRunList=input, ThetaIn=theta_in, SummationType='SumInQ', ReductionType='DivergentBeam', 
                        AnalysisMode='MultiDetectorAnalysis', 
                        ProcessingInstructions='"""+ROI+"""', 
                        WavelengthMin=1.5, WavelengthMax=17, 
                        I0MonitorIndex=2, MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18, 
                        MonitorIntegrationWavelengthMin=4, MonitorIntegrationWavelengthMax=10, SubtractBackground=True, 
                        BackgroundCalculationMethod='AveragePixelFit', 
                        FirstTransmissionRunList=trans, StartOverlap=10, EndOverlap=12, ScaleRHSWorkspace=False, 
                        TransmissionProcessingInstructions='"""+TRANS_ROI+"""', Debug=True, 
                        MomentumTransferMin=qmin, MomentumTransferStep="""+str(dq_q)+""", 
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


xd, yd, ed = np.loadtxt('text.csv', delimiter=' ', usecols=(0, 1, 2), unpack=True)
# fig = go.Figure(data=[go.Scatter(x=xd, y=yd, error_y=dict(
#         type='data',
#         array=ed,
#         visible=True))])


#
# ReflectometryISISLoadAndProcess(InputRunList='65272', ThetaIn=0.8,
#                                 AnalysisMode='MultiDetectorAnalysis', ProcessingInstructions='70-90',
#                                 WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2,
#                                 MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18, MonitorIntegrationWavelengthMin=4,
#                                 MonitorIntegrationWavelengthMax=10,
#                                 FirstTransmissionRunList='65274', SecondTransmissionRunList='65275',
#                                 StartOverlap=10, EndOverlap=12, ScaleRHSWorkspace=False,
#                                 TransmissionProcessingInstructions='70-90',
#                                 MomentumTransferMin=0.010321317306126728,
#                                 MomentumTransferStep=0.055433662337842131,
#                                 MomentumTransferMax=0.1168874036214391,
#                                 OutputWorkspaceBinned='IvsQ_binned_65272',
#                                 OutputWorkspace='IvsQ_65272',
#                                 OutputWorkspaceTransmission='TRANS_LAM_65274_65275')
#
# ReflectometryISISLoadAndProcess(InputRunList='65273', ThetaIn=2.3, AnalysisMode='MultiDetectorAnalysis', ProcessingInstructions='67-95', WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2, MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18, MonitorIntegrationWavelengthMin=4, MonitorIntegrationWavelengthMax=10, FirstTransmissionRunList='65276', SecondTransmissionRunList='65277', StartOverlap=10, EndOverlap=12, ScaleRHSWorkspace=False, TransmissionProcessingInstructions='70-90', MomentumTransferMin=0.029666234509808882, MomentumTransferStep=0.055446760622640492, MomentumTransferMax=0.33612056568876092, OutputWorkspaceBinned='IvsQ_binned_65273', OutputWorkspace='IvsQ_65273', OutputWorkspaceTransmission='TRANS_LAM_65276_65277')
# Stitch1DMany(InputWorkspaces='IvsQ_65272,IvsQ_65273', OutputWorkspace='IvsQ_65272_65273', Params='-0.055434', OutScaleFactors='0.841361')
#

# ReflectometryISISLoadAndProcess(InputRunList=wksp, ThetaIn=2.3,
#                                 AnalysisMode='MultiDetectorAnalysis', ProcessingInstructions='70-90',
#                                 WavelengthMin=1.5, WavelengthMax=17, I0MonitorIndex=2,
#                                 MonitorBackgroundWavelengthMin=17, MonitorBackgroundWavelengthMax=18,
#                                 MonitorIntegrationWavelengthMin=4, MonitorIntegrationWavelengthMax=10,
#                                 StartOverlap=10, EndOverlap=12,
#                                 MomentumTransferMin=0.0064510027925337703,
#                                 MomentumTransferStep=0.029555884192578895, MomentumTransferMax=0.073099540677375993,
#                                 OutputWorkspaceBinned='IvsQ_binned_62098', OutputWorkspace='IvsQ_62098')

# Generate figure
# xd = mtd['IvsQ_65272_65273'].dataX(0)
# yd = mtd['IvsQ_65272_65273'].dataY(0)

# xd = mtd['0_IvsQ_binned'].dataX(0)
# yd = mtd['0_IvsQ_binned'].dataY(0)

# fig = go.Figure(data=[go.Scatter(x=xd, y=yd, error_y=dict(
#             type='data',
#             array=ed,
#             visible=True))])
# fig.update_yaxes(type="log")
# fig.update_xaxes(type="log")
#
#
# fig.update_layout(
#     margin=dict(l=20, r=20, t=20, b=20),
#     paper_bgcolor="LightSteelBlue",
# )


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


ISISJournalGetExperimentRuns(Cycle='22_1', InvestigationId=values['rbno'].strip(), OutputWorkspace='RB'+values['rbno'].strip())
runs = mtd['RB'+values['rbno'].strip()]

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.CYBORG]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# Live data figure
# @app.callback(Output('reflectivity-graph', 'figure'),
#               Input('interval-component', 'n_intervals'))


def drawFigure():
    return html.Div([
        dcc.Graph(id='reflectivity-graph', figure=fig)
        # dbc.Card(dbc.CardBody([dcc.Graph(id='reflectivity-graph', figure=fig)]))
    ])


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

degree_sign = u'\N{DEGREE SIGN}'

def create_card(title, content, color):
    card_content = content.split("|")

    cb = [html.H4(title, className="card-header"), html.Br()]
    for c in card_content:
        cb.append(html.H3(c, className="card-text"))
        cb.append(html.Br())
    card = dbc.Card(
        dbc.CardBody(cb),
        color=color, inverse=False
    )
    return card


# Trans number input:
trans1_input = html.Div(
    [
        dbc.Label("Trans 1",  style={'font-size':'20px'}),
        dbc.Input(id="input1", placeholder="Transmission run number.", type="number", style={'font-size':'20px'}),
        html.Br(),
        # html.P(id="output1"),
    ]
)


@app.callback(Output("output1", "children"), [Input("input1", "value")])
def output_trans1(value):
    return value


trans2_input = html.Div(
    [
        dbc.Label("Trans 2",  style={'font-size':'20px'}),
        dbc.Input(id="input2", placeholder="Transmission run number.", type="number",  style={'font-size':'20px'}),
        html.Br(),
        # html.P(id="output2"),
    ]
)


@app.callback(Output("output2", "children"), [Input("input2", "value")])
def output_trans2(value):
    return value


def create_table(title, rows):
    table_head = [
            html.Thead(html.Tr(html.Th(html.H4(title), colSpan=str(len(rows[0])))))
        ]

    table_rows = []
    for row in rows:
        table_rows.append(html.Tr([html.Td(html.H4(text)) for text in row]))

    table_body = [html.Tbody(table_rows)]

    sample_table = dbc.Table(table_head + table_body,
                             borderless=True,
                             dark=True,
                             hover=True,
                             responsive=True,
                             striped=True,
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
                         dbc.Col(id='detector_table', children=[detector_table], md=2),],
                        style={'padding': 10})
    return graphRow1

xd, yd, ed = np.loadtxt('text.csv', delimiter=' ', usecols=(0, 1, 2), unpack=True)
fig = go.Figure(data=[go.Scatter(x=xd, y=yd, error_y=dict(
            type='data',
            array=ed,
            visible=True))])

@app.callback(Output('graph_row', 'children'),
              Input('pandas-dropdown-2', 'value'),
              Input('interval-component', 'n_intervals'))
def graph_row(value, n):
    global fig
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
            if 'IvsQ_binned_'+str(run) not in mtd.getObjectNames():
                ReflectometryISISLoadAndProcess(InputRunList=str(run), AnalysisMode='MultiDetectorAnalysis',
                                        ProcessingInstructions='70-90', OutputWorkspaceBinned='IvsQ_binned_'+str(run))
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

    graphRow2 = dbc.Row([dbc.Col(trans1_input, md=1), dbc.Col(trans2_input, md=1),
             dbc.Col(dcc.Graph(id='reflectivity-graph', figure=fig), md=6, width={"offset": 1})],
                         # dbc.Col(id='wksp_dropdown', children=[wksp_dropdown], md=2)],
              # dbc.Col(dcc.Dropdown(id="runlist-drowpdown", options=[item for item in runs.column(1)], multi=True), md=2)],
             style={'padding': 10})
    gr = dcc.Graph(id='reflectivity-graph', figure=fig)
    return gr #graphRow2

# dbc.Col(id='wksp_dropdown', children=[wksp_dropdown], md=2)],

# wksp_dropdown = dbc.DropdownMenu(
#     label="Menu",
#     children=[
#         dbc.DropdownMenuItem(item) for item in runs.column(1)
#     ],
#     color="secondary"
# )
wksp_dropdown = dcc.Dropdown(id="runlist-drowpdown", options=[item for item in runs.column(1)], multi=True)


# graphRow2 = dbc.Row([dbc.Col(trans1_input, md=1), dbc.Col(trans2_input, md=1),
#                      dbc.Col(dcc.Graph(id='reflectivity-graph', figure=fig), md=6, width={"offset": 1}),
#                      dbc.Col(id='wksp_dropdown', children=[wksp_dropdown], md=2)], style={'padding': 10})
# #################
# graphRow2 = dbc.Row([dbc.Col(trans1_input, md=1), dbc.Col(trans2_input, md=1),
#                      dbc.Col(dcc.Graph(id='reflectivity-graph', figure=fig), md=6, width={"offset": 1}),
#                      dbc.Col(
#                          dcc.Dropdown(id="runlist-drowpdown", options=[item for item in runs.column(1)], multi=True), md=2)],
#                     style={'padding': 10})

app.layout = html.Div([html.Div(id='row0'),
                       html.Div(id='row1'),
                       # html.Div(id='graph_row', children=[]),#graphRow2,
                       dbc.Row([dbc.Col(id="trans_col", children=[dbc.Row(trans1_input), dbc.Row(trans2_input)], md=1),
                                # dbc.Col(trans2_input, md=1),
                                dbc.Col(html.Div(id='graph_row'), md=7, width={"offset": 1}),
                                dbc.Col(dcc.Dropdown([item for item in runs.column(1)], id='pandas-dropdown-2',
                                                     multi=True,  style={'font-size':'20px'}), md=2)],
                                style={'padding': 10}),
                       # html.Div(id='pandas-output-container-2'),
                       dcc.Interval(
                            id='interval-component',
                            interval=5*1000, # in milliseconds
                            n_intervals=0
                      )]
                      )


# @app.callback(
#     Output('reflectivity-graph', 'figure'),
#     # Output('pandas-output-container-2', 'children'),
#     Input('pandas-dropdown-2', 'value')
# )
def update_output(value):
    print("Processing: ", value)

    ReflectometryISISLoadAndProcess(InputRunList=str(value), AnalysisMode='MultiDetectorAnalysis',
                                    ProcessingInstructions='70-90', OutputWorkspaceBinned='IvsQ_binned_'+str(value))
    # xd, yd, ed = np.loadtxt('text.csv', delimiter=' ', usecols=(0, 1, 2), unpack=True)

    # fig = go.Figure(data=[go.Scatter(x=xd, y=yd, error_y=dict(
    #     type='data',
    #     array=ed,
    #     visible=True))])
    # fig.update_yaxes(type="log", tickfont_size=18)
    # fig.update_xaxes(type="log", tickfont_size=18)
    #
    # fig.update_layout(
    #     yaxis_tickformat='.0e',
    #     uirevision="Don't change",
    #     margin=dict(l=20, r=20, t=20, b=20),
    #     paper_bgcolor="LightSteelBlue",
    # )
    # fig.update_yaxes(nticks=6)

    xd = mtd['IvsQ_binned_'+str(value)].dataX(0)
    yd = mtd['IvsQ_binned_'+str(value)].dataY(0)
    ed = mtd['IvsQ_binned_'+str(value)].dataE(0)

    fig.add_trace(go.Scatter(x=xd, y=yd, error_y=dict(
        type='data',
        array=ed,
        visible=True)))
    # fig.add_trace(go.Scatter(x=xd, y=yd))

    return fig #f'You have selected {value}'




if __name__ == '__main__':
    # t1.start()
    # print(xd)
    app.run_server(host="0.0.0.0", port="8050")