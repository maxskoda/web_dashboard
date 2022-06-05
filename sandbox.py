import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State

from mantid.simpleapi import *
import matplotlib.pyplot as plt

ISISJournalGetExperimentRuns(Cycle='22_1', InvestigationId='2210016', OutputWorkspace='RB2210016')
runs = mtd['RB2210016']
opt = [item for item in runs.column(1)]

app = dash.Dash()

app.layout = html.Div(
    [
        dcc.Interval(id="interval", interval=3000),
        dcc.Checklist(
            id="checklist",
            options=[
                {"label": "value 1", "value": 1},
                {"label": "value 2", "value": 2},
                {"label": "value 3", "value": 3},
            ],
            value=[1],
        ),
        dcc.Dropdown(id="dropdown"),
    ]
)


@app.callback(
    [Output("dropdown", "options"), Output("dropdown", "value")],
    [Input("interval", "n_intervals")],
    [State("dropdown", "value"), State("checklist", "value")]
)
def make_dropdown_options(n, value, values):
    options = [{"label": f"Option {v}", "value": v} for v in opt]

    if value not in [o["value"] for o in options]:
        # if the value is not in the new options list, we choose a different value
        if options:
            value = options[0]["value"]
        else:
            value = None
    print([{"label": f"Option {v}", "value": v} for v in opt])
    return options, value


if __name__ == "__main__":
    app.run_server(debug=True)