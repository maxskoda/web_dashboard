import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash()

app.layout = html.Div(
    [
        dcc.Interval(id="interval"),
        html.P(id="output"),
        html.Button("toggle interval", id="button"),
    ]
)


@app.callback(
    Output("interval", "disabled"),
    [Input("button", "n_clicks")],
    [State("interval", "disabled")],
)
def toggle_interval(n, disabled):
    print(n)
    if n:
        return not disabled
    return disabled


@app.callback(Output("output", "children"), [Input("interval", "n_intervals")])
def display_count(n):
    return f"Interval has fired {n} times"


if __name__ == "__main__":
    app.run_server(debug=True)