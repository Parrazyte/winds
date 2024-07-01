import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
def gaussian(x, mu=0, sig=1):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


steprange=np.linspace(1,3,21)
xrange=np.arange(-1.5, 1.5, 0.04)
y1=[50*i*gaussian(xrange,mu=1,sig=i) for i in steprange]



def EW_25(x,g):
    norm = 50
    width = g / 4
    return norm*np.sqrt(2*np.pi)*(width)*g/2*gaussian(x,mu=1.5*width-1.,sig=width)

def EW_26(x,g):

    norm = 15
    width = g/4
    # width = 0.25+g/6

    return norm*np.sqrt(2*np.pi)*(width)*gaussian(x,mu=2*width-1,sig=width)*(1+g/30)

def U25(EW25,g):
    return [3/8*g-1-np.sqrt(-g**2/8*np.log(EW25/(25*g))),3/8*g-1+np.sqrt(-g**2/8*np.log(EW25/(25*g)))]

def U26(EW26,g):
    return [g/2-1-np.sqrt(-g**2/8*np.log(EW26/(15))),3/8*g-1+np.sqrt(-g**2/8*np.log(EW26/(15)))]


def Usingle(EW25,EW26,g):
    return 7*g/16-1-g/2*np.log(EW25/EW26*3/(5*g)*(1+g/30))


fig = go.Figure()
fig = make_subplots(rows=2, cols=3)
for i_step,step in enumerate(steprange):

    xrange3=xrange+(1.5*step/4-1.)
    xrange4=xrange+(2*(step/4)-1)
    # xrange4=xrange+(2*(0.25+step/6)-1)

    x3b=xrange/(2+step**2)


    xrange3b=xrange3*3/4

    y1=EW_25(xrange,step)
    y2=EW_26(xrange,step)

    U_invert=Usingle(y1,y2,step)

    # breakpoint()
    y3=EW_25(xrange3,step)/step
    y3b=EW_25(xrange,step)/step
    y4=EW_26(xrange4,step)
    y4b=EW_26(xrange,step)
    r1=y1/y2
    r2=y3b/y4b

    fig.add_trace(
        go.Scatter(
            visible=False,
            x=xrange,
            y=y1),row=1, col=1)

    fig.add_trace(
        go.Scatter(
            visible=False,
            x=xrange,
            y=y2),row=1, col=2)

    fig.add_trace(
        go.Scatter(
            visible=False,
            x=xrange,
            y=y1/y2),row=1, col=3)

    fig.add_trace(
        go.Scatter(
            visible=False,
            x=xrange,
            y=y3),row=2, col=1)

    fig.add_trace(
        go.Scatter(
            visible=False,
            x=x3b,
            y=y3b,line=dict(dash='dash')),row=2, col=1)


    fig.add_trace(
        go.Scatter(
            visible=False,
            x=xrange,
            y=y4),row=2, col=2)

    fig.add_trace(
        go.Scatter(
            visible=False,
            x=x3b,
            y=y4b,line=dict(dash='dash')),row=2, col=2)

    fig.add_trace(
        go.Scatter(
            visible=False,
            x=xrange,
            y=y3/y4),row=2, col=3)

    fig.add_trace(
        go.Scatter(
            visible=False,
            x=U_invert,
            y=y1/y2),row=2, col=3)




# Create and add slider
steps = []

ngraphs=9

for i in range(ngraphs):
    fig.data[20+i].visible = True

for i in range(len(fig.data)):
    if not i%ngraphs==0:
        continue
    step = dict(
        method="update",
        args=[
            {"visible": [False] * ngraphs*len(fig.data)},
            {"title": "Slider switched to step: " + str(i)},
            {'traces':[0,1,2,3,4,5,6,7,8,9]}],
    )

    for j in range(5):
        step["args"][0]["visible"][min(ngraphs*(i//ngraphs)+ngraphs*j,len(fig.data))] = True
        step["args"][0]["visible"][min(ngraphs * (i // ngraphs) +1 + ngraphs * j, len(fig.data))] = True
        step["args"][0]["visible"][min(ngraphs * (i // ngraphs) +2 + ngraphs * j, len(fig.data))] = True
        step["args"][0]["visible"][min(ngraphs * (i // ngraphs) +3 + ngraphs * j, len(fig.data))] = True
        step["args"][0]["visible"][min(ngraphs * (i // ngraphs) +4 + ngraphs * j, len(fig.data))] = True
        step["args"][0]["visible"][min(ngraphs * (i // ngraphs) +5 + ngraphs * j, len(fig.data))] = True
        step["args"][0]["visible"][min(ngraphs * (i // ngraphs) +6 + ngraphs * j, len(fig.data))] = True
        step["args"][0]["visible"][min(ngraphs * (i // ngraphs) +7 + ngraphs * j, len(fig.data))] = True
        step["args"][0]["visible"][min(ngraphs * (i // ngraphs) +8 + ngraphs * j, len(fig.data))] = True

    steps.append(step)

# breakpoint()
sliders = [dict(
    active=20,
    currentvalue={"prefix": "gamma: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(sliders=sliders)

yax1=list((fig.select_yaxes(row=1,col=1)))[0]
yax1.update(range=[0,60])

yax3=list((fig.select_yaxes(row=2,col=1)))[0]
yax3.update(range=[0,60])


yax2=list((fig.select_yaxes(row=1,col=2)))[0]
yax2.update(range=[0,20])

yax4=list((fig.select_yaxes(row=2,col=2)))[0]
yax4.update(range=[0,20])

yaxr1=list((fig.select_yaxes(row=1,col=3)))[0]
yaxr1.update(range=[-3,2],type="log")

yaxr2=list((fig.select_yaxes(row=2,col=3)))[0]
yaxr2.update(range=[-3,2],type="log")

st.plotly_chart(fig,use_container_width=True)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig2= go.Figure()
fig2 = make_subplots(rows=2, cols=3)
for i_step,step in enumerate(steprange):

    y1=EW_25(xrange,step)
    y2=EW_26(xrange,step)

