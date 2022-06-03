import dash
from dash.dependencies import Input, Output
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

app.layout = html.Div([
    daq.Gauge(
        id='my-gauge-1',
        label="Default",
        value=6
    ),
    dcc.Slider(
        id='my-gauge-slider-1',
        min=10,
        max=0,
        step=-1,
        value=5
    ),
])

@app.callback(Output('my-gauge-1', 'value'), Input('my-gauge-slider-1', 'value'))
def update_output(value):
    return value

if __name__ == '__main__':
    app.run_server(debug=True)
