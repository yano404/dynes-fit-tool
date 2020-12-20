import base64
import io
import json

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


# dynes function
def dynes(E, delta, gamma, C=0.0, offset=0.0):
    igam = gamma * 1.0j
    return (
        C
        * np.abs(
            np.real((E - igam) / np.power(np.square(E - igam) - np.square(delta), 0.5))
        )
        + offset
    )


# setup app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# function to get file extention
def get_ext(filename):
    return filename.split(".")[-1]


@app.callback(
    [
        Output("data-table-name", "children"),
        Output("data-table", "data"),
        Output("data-table", "columns"),
        Output("data-table", "filter_action"),
    ],
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")],
)
def update_table(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        ext = get_ext(filename)
        try:
            if ext == "csv":
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            elif ext == "xls" or ext == "xlsx":
                df = pd.read_excel(io.BytesIO(decoded))
            elif ext == "txt":
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
            else:
                return (
                    dbc.Alert(
                        [
                            html.H5("Error", className="alert-heading"),
                            html.P("This file type is not supported."),
                        ],
                        color="danger",
                    ),
                    None,
                    None,
                    "none",
                )
        except Exception as e:
            print(e)
            return (
                dbc.Alert(
                    [
                        html.H5("Error", className="alert-heading"),
                        html.P("There was an error processing this file."),
                    ],
                    color="danger",
                ),
                None,
                None,
                "none",
            )
        voltage = df.loc[:, df.columns.str.contains("voltage")]
        current = df.loc[:, df.columns.str.contains("current")]
        if voltage.size and current.size:
            conductance = np.gradient(current.squeeze(), voltage.squeeze())
            df["conductance"] = conductance / conductance[-1]
        return (
            html.H6(filename, style={"word-break": "break-all"}),
            df.to_dict("records"),
            [{"name": i, "id": i} for i in df.columns],
            "native",
        )
    else:
        return (
            html.H6("filename", style={"word-break": "break-all"}),
            None,
            None,
            "none",
        )


@app.callback(
    [Output("graph-xaxis", "options"), Output("graph-yaxis", "options")],
    [Input("data-table", "columns")],
)
def update_graph_axis_select(columns):
    if not columns:
        raise PreventUpdate
    options = [{"label": c["name"], "value": c["id"]} for c in columns]
    return (options, options)


@app.callback(
    Output("fit-range-indicator", "children"), [Input("fit-range-slider", "value")]
)
def display_fit_range(value):
    if value is not None:
        return html.P(f"{value[0]} : {value[1]}")


@app.callback(
    [
        Output("fit-range-min", "value"),
        Output("fit-range-max", "value"),
        Output("apply-fit-range-min-max", "n_clicks"),
    ],
    [
        Input("data-table", "data"),
        Input("data-table", "columns"),
        Input("graph-xaxis", "value"),
        Input("graph-yaxis", "value"),
    ],
)
def update_fit_range_min_max(rows, columns, xaxis_id, yaxis_id):
    if not rows or not xaxis_id or not yaxis_id:
        raise PreventUpdate
    df = pd.DataFrame(rows, columns=[c["id"] for c in columns])
    if not (xaxis_id in df.columns) or not (yaxis_id in df.columns):
        raise PreventUpdate
    xmin = df.min()[xaxis_id]
    xmax = df.max()[xaxis_id]
    return (xmin, xmax, 0)


@app.callback(
    [
        Output("fit-range-slider", "min"),
        Output("fit-range-slider", "max"),
        Output("fit-range-slider", "value"),
    ],
    [Input("apply-fit-range-min-max", "n_clicks")],
    [
        State("fit-range-min", "value"),
        State("fit-range-max", "value"),
        State("fit-range-slider", "value"),
    ],
)
def update_fit_range(apply_clicked, xmin, xmax, fit_range):
    if xmin is None or xmax is None:
        raise PreventUpdate
    if xmin >= xmax:
        raise PreventUpdate
    if fit_range:
        fit_range_lower, fit_range_upper = fit_range
        if fit_range_lower <= xmin:
            fit_range_lower = xmin
        if fit_range_upper >= xmax:
            fit_range_upper = xmax
        value = [fit_range_lower, fit_range_upper]
    else:
        value = [xmin, xmax]
    return (xmin, xmax, value)


@app.callback(
    [Output("graph", "figure"), Output("fit-result", "children")],
    [
        Input("data-table", "data"),
        Input("data-table", "columns"),
        Input("graph-xaxis", "value"),
        Input("graph-yaxis", "value"),
        Input("fit-button", "n_clicks"),
    ],
    [
        State("fit-range-slider", "value"),
        State("plot-fit-range", "value"),
        State("use-bounds", "value"),
        State("D-lower", "value"),
        State("G-lower", "value"),
        State("C-lower", "value"),
        State("offset-lower", "value"),
        State("D-upper", "value"),
        State("G-upper", "value"),
        State("C-upper", "value"),
        State("offset-upper", "value"),
    ],
)
def update_graph(
    rows,
    columns,
    xaxis_id,
    yaxis_id,
    n_clicks,
    fit_range,
    plot_fit_range,
    use_bounds,
    D_low,
    G_low,
    C_low,
    offset_low,
    D_up,
    G_up,
    C_up,
    offset_up,
):
    """
    rows : [
        {'column_A': value_A_1, 'column_B': value_B_1, ...},
        ...
    ]
    columns : [
        {'name': 'column_A_name', 'id': 'column_A_id'},
        {'name': 'column_B_name', 'id': 'column_B_id'},
        ...
    ]
    fit_range : [min, max]
    use_bounds : if True: [True]; else: [];
    """
    ctx = dash.callback_context
    if not rows or not xaxis_id or not yaxis_id:
        raise PreventUpdate
    df = pd.DataFrame(rows, columns=[c["id"] for c in columns])
    if not (xaxis_id in df.columns) or not (yaxis_id in df.columns):
        raise PreventUpdate
    fig = go.Figure(
        data=[
            go.Scatter(x=df[xaxis_id], y=df[yaxis_id], mode="markers", name="observed")
        ],
        layout=go.Layout(
            xaxis_title=str(xaxis_id),
            yaxis_title=str(yaxis_id),
            template="plotly_white",
            showlegend=False,
            uirevision=xaxis_id,
        ),
    )
    # if fit-button pushed: run scipy.optimize.curve_fit and plot function
    if ctx.triggered[0]["prop_id"] == "fit-button.n_clicks":
        if use_bounds:
            lower_bounds = np.array([D_low, G_low, C_low, offset_low])
            upper_bounds = np.array([D_up, G_up, C_up, offset_up])
            if None in lower_bounds or None in upper_bounds:
                raise PreventUpdate
            if np.any(lower_bounds >= upper_bounds):
                raise PreventUpdate
            bounds = (lower_bounds, upper_bounds)
        else:
            bounds = ((1000.0, 0.0, 0.1, -1.0), (5000.0, 2000.0, 2.0, 1.0))
        # fit
        lower_lim, upper_lim = fit_range
        df_range = df[(lower_lim < df[xaxis_id]) & (df[xaxis_id] < upper_lim)]
        popt, pcov = scipy.optimize.curve_fit(
            dynes, df_range[xaxis_id], df_range[yaxis_id], bounds=bounds
        )
        perr = np.sqrt(np.diag(pcov))
        # plot function
        xmin = df.min()[xaxis_id]
        xmax = df.max()[xaxis_id]
        x = np.linspace(xmin, xmax, 200)
        fig.add_trace(go.Scatter(x=x, y=dynes(x, *popt), mode="lines", name="fit"))
        # plot fit range
        if plot_fit_range:
            fig.add_vrect(
                x0=lower_lim,
                x1=upper_lim,
                fillcolor="LightGray",
                opacity=0.3,
                layer="below",
                line_width=0,
            )

        # show popt, perr
        tab_header = [
            html.Thead(html.Tr([html.Th("param"), html.Th("value"), html.Th("err")]))
        ]
        tab_body = [
            html.Tbody(
                [
                    html.Tr([html.Td(p), html.Td(v), html.Td(e)])
                    for p, v, e in zip(
                        ["Delta", "Gamma", "Const", "Offset"], popt, perr
                    )
                ]
            )
        ]
        result = html.Div(
            [
                dbc.Table(tab_header + tab_body),
                html.Hr(),
                html.Pre(
                    json.dumps(
                        {
                            "popt": popt.tolist(),
                            "perr": perr.tolist(),
                            "pcov": pcov.tolist(),
                        },
                        indent=2,
                    )
                ),
            ]
        )
    else:
        result = None

    return (fig, result)


# UI components
# header
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
header = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Dynes Fit", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="#",
        )
    ],
    color="dark",
    dark=True,
    fixed="top",
    sticky="top",
)

# file uploader
file_upload = html.Div(
    [
        html.H4(
            className="mx-1 mt-3",
            children=["upload file"],
            style={"textAlign": "center"},
        ),
        dcc.Upload(
            id="upload-data",
            className="mx-1",
            children=html.Div(
                [
                    "Drag and Drop or ",
                    html.A("Select Files"),
                ]
            ),
            # Disable multiple files upload
            multiple=False,
        ),
    ]
)

# data table
data_table = html.Div(
    id="data-table-container",
    className="mx-1 mt-3",
    children=[
        # loading state
        dcc.Loading(
            id="loading-file",
            type="circle",
            children=html.Div(id="data-table-name"),
        ),
        dash_table.DataTable(
            id="data-table",
            row_deletable=True,
            # filter_action='native',
            fixed_rows={"headers": True},
            page_action="none",
            style_table={"height": "auto", "overflowY": "auto"},
        ),
    ],
)

# sidebar
sidebar = html.Div(
    id="sidebar",
    children=[
        file_upload,
        html.Hr(),
        data_table,
        html.Hr(),
    ],
)

# graph
graph = dbc.Card(
    id="graph-card",
    children=[
        dbc.CardHeader("Graph"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [html.H6("x-axis"), dcc.Dropdown(id="graph-xaxis")]
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            html.Div(
                                [html.H6("y-axis"), dcc.Dropdown(id="graph-yaxis")]
                            ),
                            width=6,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="graph", style={"height": "65vh"}  # temporary set
                            ),
                            width=12,
                        )
                    ]
                ),
            ],
            style={"height": "80vh"},
        ),
    ],
)

# fit panel
fit_panel = dbc.Card(
    id="fit-panel",
    children=[
        dbc.CardHeader("Fit Panel"),
        dbc.CardBody(
            [
                dbc.Button(
                    "Fit",
                    id="fit-button",
                    color="primary",
                    block=True,
                    className="mb-3",
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Fit Range", html_for="fit-range-slider"),
                        dcc.RangeSlider(id="fit-range-slider"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Input(
                                        id="fit-range-min",
                                        type="number",
                                        bs_size="sm",
                                    ),
                                    width=5,
                                ),
                                dbc.Col(
                                    dbc.Input(
                                        id="fit-range-max",
                                        type="number",
                                        bs_size="sm",
                                    ),
                                    width=5,
                                ),
                            ],
                            justify="between",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Checklist(
                                        id="plot-fit-range",
                                        options=[
                                            {"label": "Display range", "value": True}
                                        ],
                                        value=[],
                                        switch=True,
                                    ),
                                    width=7,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Apply",
                                        id="apply-fit-range-min-max",
                                        size="sm",
                                        color="primary",
                                        block=True,
                                    ),
                                    width=5,
                                ),
                            ],
                            align="center",
                            justify="between",
                            className="mt-1",
                        ),
                    ]
                ),
                html.Div(id="fit-range-indicator"),
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                dbc.Checklist(
                                    id="use-bounds",
                                    options=[{"label": "Bounds", "value": True}],
                                    value=[],
                                    switch=True,
                                )
                            ]
                        ),
                        dbc.CardBody(
                            [
                                html.H6("Delta", className="text-center"),
                                dbc.FormGroup(
                                    [
                                        dbc.Label("lower bounds", html_for="D-lower"),
                                        dbc.Input(
                                            id="D-lower",
                                            placeholder="lower bounds",
                                            type="number",
                                            bs_size="sm",
                                        ),
                                        dbc.Label("upper bounds", html_for="D-upper"),
                                        dbc.Input(
                                            id="D-upper",
                                            placeholder="upper bounds",
                                            type="number",
                                            bs_size="sm",
                                        ),
                                    ]
                                ),
                                html.H6("Gamma", className="text-center"),
                                dbc.FormGroup(
                                    [
                                        dbc.Label("lower bounds", html_for="G-lower"),
                                        dbc.Input(
                                            id="G-lower",
                                            placeholder="lower bounds",
                                            type="number",
                                            bs_size="sm",
                                        ),
                                        dbc.Label("upper bounds", html_for="G-upper"),
                                        dbc.Input(
                                            id="G-upper",
                                            placeholder="upper bounds",
                                            type="number",
                                            bs_size="sm",
                                        ),
                                    ]
                                ),
                                html.H6("Const", className="text-center"),
                                dbc.FormGroup(
                                    [
                                        dbc.Label("lower bounds", html_for="C-lower"),
                                        dbc.Input(
                                            id="C-lower",
                                            placeholder="lower bounds",
                                            type="number",
                                            bs_size="sm",
                                        ),
                                        dbc.Label("upper bounds", html_for="C-upper"),
                                        dbc.Input(
                                            id="C-upper",
                                            placeholder="upper bounds",
                                            type="number",
                                            bs_size="sm",
                                        ),
                                    ]
                                ),
                                html.H6("Offset", className="text-center"),
                                dbc.FormGroup(
                                    [
                                        dbc.Label(
                                            "lower bounds", html_for="offset-lower"
                                        ),
                                        dbc.Input(
                                            id="offset-lower",
                                            placeholder="lower bounds",
                                            type="number",
                                            bs_size="sm",
                                        ),
                                        dbc.Label(
                                            "upper bounds", html_for="offset-upper"
                                        ),
                                        dbc.Input(
                                            id="offset-upper",
                                            placeholder="upper bounds",
                                            type="number",
                                            bs_size="sm",
                                        ),
                                    ]
                                ),
                            ],
                            className="overflow-auto",
                            style={"height": "35vh"},
                        ),
                    ]
                ),
                dbc.Card(
                    [
                        dbc.CardHeader([html.H6("Result")]),
                        dbc.CardBody(
                            [html.Div(id="fit-result")],
                            className="overflow-auto",
                            style={"height": "20vh"},
                        ),
                    ],
                    className="mt-3",
                ),
            ],
            className="overflow-auto",
            style={"height": "80vh"},
        ),
    ],
)

# layout
app.layout = html.Div(
    [
        html.Div([sidebar]),
        header,
        html.Div(
            id="content",
            children=[dbc.Row([dbc.Col(graph, width=8), dbc.Col(fit_panel, width=4)])],
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
