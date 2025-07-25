import json
import dash
from dash import html, dcc, Input, Output, State, clientside_callback
import plotly.graph_objects as go
import dill

import getpass
username=getpass.getuser()

'''
Loads a 3D plot saved with dill 
and gives an interactive view of the camera layout 
largely taken from https://community.plotly.com/t/how-to-obtain-camera-variables-after-manually-changing-them/86297
'''


plot_path='/home/'+username+'/Documents/Work/PhD/docs/papers/wind_4U/global/ion_visu/save_figs/fig_3D.pkl'

with open(plot_path,'rb') as f:
    fig=dill.load(f)

fig.update_layout(width=1000)

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css",
    ]
)

app.layout = html.Div(
    [
        html.Div([
            html.Div(
                [
                    dcc.Graph(id='graph', figure=fig),
                    html.Button(id='iso', children='apply'),
                ],
                className='vstack'
            ),
            html.Div(
                [
                    html.H5('Eye vector', className='me-auto text-nowrap pe-5'),
                    dcc.Input(id='eye_x', type='number', value=1.8),
                    dcc.Input(id='eye_y', type='number', value=1.8),
                    dcc.Input(id='eye_z', type='number', value=1.8)
                ],
                className='hstack'

            ),
            html.Div(
                [
                    html.H5('Up vector', className='me-auto text-nowrap pe-5'),
                    dcc.Input(id='up_x', type='number', value=0.),
                    dcc.Input(id='up_y', type='number', value=0.),
                    dcc.Input(id='up_z', type='number', value=1.),
                ],
                className='hstack'
            ),
        ], className='col'
        ),
        html.Div(
            [
                html.Pre(id='out'),
            ],
            className='col'
        ),
        html.Div(
            [
                html.Pre(id='out2'),
            ],
            className='col'
        ),
        html.Div(
            [
                html.Pre(id='out3'),
            ],
            className='col'
        )
    ],
    className='row'
)


clientside_callback(
    """
    function(click, a, b, c, d, e, f, fig) {
        const newFig = JSON.parse(JSON.stringify(fig));
        var newView = {
           // 'layout': {
           //     'scene': {
                    // 'aspectmode': 'data',
                    'camera': {
                        'eye': {'x': a, 'y': b, 'z': c},
                        'up': {'x': d, 'y': e, 'z': f},
                        'projection': {'type': 'orthographic'}
                    }
                };
           // };
        //};
        const newLayout = Object.assign(newFig.layout.scene, newView);
        newFig['layout']['scene'] = newLayout;
        return newFig;
    }
    """,
    Output('graph', 'figure'),
    Input('iso', 'n_clicks'),
    State('eye_x', 'value'),
    State('eye_y', 'value'),
    State('eye_z', 'value'),
    State('up_x', 'value'),
    State('up_y', 'value'),
    State('up_z', 'value'),
    State('graph', 'figure'),
    prevent_initial_call=True
)


@app.callback(
    Output('out', 'children'),
    # Output('out2', 'children'),
    Output('out3', 'children'),
    Input('graph', 'relayoutData'),
    # Input('graph', 'restyleData'),
    Input('graph', 'figure'),
    prevent_initial_call=True
)

def update(relayoutData, current_figure):
    return [
        html.Div([html.H5('Figure data'), json.dumps(current_figure['layout'], indent=3)]),
        html.Div([html.H5('relayout data'), json.dumps(relayoutData, indent=3)])]

# def update(relayoutData, restyleData, current_figure):
#     return [
#         html.Div([html.H5('Figure data'), json.dumps(current_figure['layout'], indent=3)]),
#         html.Div([html.H5('relayout data'), json.dumps(relayoutData, indent=3)]),
#         html.Div([html.H5('restyle data'), json.dumps(restyleData, indent=3)])
#     ]


if __name__ == "__main__":
    app.run_server(debug=True)