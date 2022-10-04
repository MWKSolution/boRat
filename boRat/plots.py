import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def model_traces(model):
    X, Y, Z = np.mgrid[-5:5:50j, -5:5:50j, -5:5:50j]
    normal_vector = model.rockNEV.dip.vector
    parallel_vector = model.wbo.vector
    # vector [A, B, C] - vector normal to bedding plane
    A, B, C = normal_vector[0], normal_vector[1], normal_vector[2]
    # vector [a, b, c] - vector parallel to borehole axis
    a, b, c = parallel_vector[0], parallel_vector[1], parallel_vector[2]
    # equation of the plane perpendicular to vector [A, B, C]
    bedding = A * X + B * Y + C * Z
    # equation of the cylinder with axis parallel to vector [a, b, c]
    well = (b * X - a * Y) ** 2 + (c * Y - b * Z) ** 2 + (a * Z - c * X) ** 2

    rock = go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=bedding.flatten(),
        opacity=0.5,
        colorscale='Earth',
        showscale=False,
        showlegend=False,
        surface=dict(count=15),
        caps=dict(x_show=False, y_show=False, z_show=False))
    well = go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=well.flatten(),
        showscale=False,
        showlegend=False,
        opacity=1,
        isomin=1.5,
        isomax=1.5,
        surface=dict(count=1),
        caps=dict(x_show=False, y_show=False, z_show=False))

    line_mode = dict(mode='lines', line=dict(color='black', width=3), showlegend=False)
    x, y, z = np.array([-5, 5]), np.array([-5, -5]), np.array([-5, -5])
    northX = go.Scatter3d(x=x, y=y, z=z, **line_mode)
    x, y, z = np.array([-5, -5]), np.array([-5, 5]), np.array([-5, -5])
    eastY = go.Scatter3d(x=x, y=y, z=z, **line_mode)
    x, y, z = np.array([-5, -5]), np.array([-5, -5]), np.array([-5, 5])
    downZ = go.Scatter3d(x=x, y=y, z=z, **line_mode)

    return [rock, well, northX, eastY, downZ]


def get_scene():
    axis_mode = dict(title_text='', showticklabels=False, showbackground=False, showaxeslabels=False, showspikes=False)
    scene = dict(
        hovermode=False,
        xaxis=axis_mode,
        yaxis=axis_mode,
        zaxis=axis_mode,
        camera=dict(
            up=dict(x=1, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.1, y=-1.1, z=-1.1)),  # x=1.25, y=1.25, z=1.25
        annotations=[dict(showarrow=False, x=5.5, y=-5, z=-5, text='X North', font=dict(size=20), opacity=0.8),
                     dict(showarrow=False, x=-5, y=5.5, z=-5, text='Y East', font=dict(size=20), opacity=0.8),
                     dict(showarrow=False, x=-5, y=-5, z=5.5, text='Z Down', font=dict(size=20), opacity=0.8)])
    return scene


def model_plot(model):
    data = model_traces(model)
    fig = go.Figure(data=data)
    fig.update_layout(scene=get_scene())
    return fig


def stress_traces(model):
    data = []
    for theta in range(361):
        hst = model.get_hoop_stress(theta).stress
        data_line = {'theta' : theta,
                     'sig_rr': hst[0, 0],
                     'sig_tt': hst[1, 1],
                     'sig_zz': hst[2, 2],
                     'tau_rt': hst[0, 1],
                     'tau_rz': hst[0, 2],
                     'tau_tz': hst[1, 2]}
        data.append(data_line)
    df = pd.DataFrame(data)

    fig_data = [go.Scatter(x=df['theta'], y=df['sig_rr'], name='sig rr', line=dict(color='blue')),
                go.Scatter(x=df['theta'], y=df['sig_tt'], name='sig \u03B8\u03B8', line=dict(color='red')),
                go.Scatter(x=df['theta'], y=df['sig_zz'], name='sig zz', line=dict(color='green')),
                go.Scatter(x=df['theta'], y=df['tau_rz'], name='tau rz', line=dict(color='orange', dash='dot')),
                go.Scatter(x=df['theta'], y=df['tau_rt'], name='tau r\u03B8', line=dict(color='cyan', dash='dot')),
                go.Scatter(x=df['theta'], y=df['tau_tz'], name='tau \u03B8z', line=dict(color='fuchsia', dash='dot'))]
    return fig_data


def stress_plot(model):
    fig = go.Figure()
    data = stress_traces(model)
    for i in data:
        fig.add_trace(i)
    fig.update_layout(yaxis=get_bedding_yaxis())
    return fig


def bedding_trace(model):
    xx = np.linspace(0, 360, num=180)
    yy = np.linspace(-2, 2, num=100)
    X, Y = np.meshgrid(np.radians(xx), yy)
    beta = np.radians(model.angle)
    wbo, dip = model.wbo, model.rockNEV.dip
    hazi = 0 if wbo.orien.hdev == 0 else wbo.orien.hazi
    ddir = 0 if dip.dip == 0 else dip.dir
    fi = np.radians(hazi - ddir)
    Z = np.sin(beta) * np.sin(X-np.pi/2+fi) - np.cos(beta) * Y
    data = go.Contour(x=xx, y=yy, z=Z,
                      colorscale='Earth',
                      showscale=False,
                      opacity=0.7,
                      hoverinfo='skip')
    return data


def get_angle_xaxis():
    return dict(dtick=45, showgrid=True, title='Angle from TOH')


def get_bedding_yaxis():
    return dict(showticklabels=False)


def bedding_plot(model):
    fig = go.Figure(bedding_trace(model))
    fig.update_layout(xaxis=get_angle_xaxis(), yaxis=get_bedding_yaxis())
    fig.update_layout(width=1000, height=1000)
    return fig


def all_plot(model):
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'type': 'scene'}, {'type': 'xy'}, {'type': 'xy'}]],
                        subplot_titles=['3D Model view', 'Unwrapped borehole wall bedding view', 'Stress tensor components'])
    fig.update_layout(scene=get_scene())

    # model subplot
    data = model_traces(model)
    for i in data:
        fig.add_trace(i,
                      row=1, col=1)
    # bedding subplot
    fig.add_trace(bedding_trace(model),
                  row=1, col=2)
    fig.update_layout(xaxis=get_angle_xaxis(), yaxis=get_bedding_yaxis())

    # stress subplot
    data = stress_traces(model)
    for i in data:
        fig.add_trace(i,
                      row=1, col=3)
    fig.update_layout(xaxis2=get_angle_xaxis())

    # overall layout
    fig.update_layout(width=1800, height=600)
    description = f'Borehole stress and bedding model:' \
                  f' far field stresses: SH: {model.stressNEV.stress[0,0]}, Sh: {model.stressNEV.stress[1,1]},' \
                  f' Sz: {model.stressNEV.stress[2,2]}, SHazi: error;' \
                  f' formation: dip: {model.rockNEV.dip.dip}, direction: {model.rockNEV.dip.dir};' \
                  f' wellbore: azi: {model.wbo.orien.hazi}, dev: {model.wbo.orien.hdev}, mud pressure: {model.wbo.Pw};' \
                  f' rock: {model.rockNEV.symmetry}'
    fig.update_layout(title=dict(text=description))
    return fig


def compare_stresses_plot(model1, model2):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[model1.Hoop.__name__, model2.Hoop.__name__])

    data1 = stress_traces(model1)
    for i in data1:
        fig.add_trace(i,
                      row=1, col=1)

    data2 = stress_traces(model2)
    for i in data2:
        fig.add_trace(i,
                      row=1, col=2)
    fig.update_layout(xaxis=get_angle_xaxis(), xaxis2=get_angle_xaxis())
    return fig
