import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import plotly_express as px
import plotly.graph_objects as pg
def Ineratctive_plot(data):
    x_axis_val=st.selectbox("Select X axis value",options=df.columns)
    y_axis_val=st.selectbox("Select Y axis value",options=df.columns)
    plot=px.scatter(data,x=x_axis_val,y=y_axis_val)
    st.plotly_chart(plot)

def Interactive(data):
 
    data["STRIKE"] = data["STRIKE"].astype(str).str.replace(",", "").str.strip()
    x_axis=st.selectbox("Select X axis Value",options=data.columns)
    y_axis=st.selectbox("Select Y axis Valie",options=data.columns)
    data[x_axis]=pd.to_numeric(data[x_axis],errors="coerce")
    data[y_axis]=pd.to_numeric(data[y_axis],errors="coerce")
    data=data.dropna(subset=[x_axis,y_axis])
    data=data.sort_values(x_axis)
    x=data[x_axis].to_numpy()
    y=data[y_axis].to_numpy()
    
    st.text(len(data))
    coeffs=np.polyfit(x,y,deg=12)
    poly_eq=np.poly1d(coeffs)
    y_fit=poly_eq(x)
    fig=pg.Figure()
    fig.add_trace(pg.Scatter(x=x,y=y,
                             mode="markers",
                             name="data"))
    fig.add_trace(pg.Scatter(x=x,y=y_fit,
                             mode="lines",
                             name="best fit 2nd degree"))
    
    
def graph(x_axis,y_axis,data):
    data["STRIKE"] = data["STRIKE"].astype(str).str.replace(",", "").str.strip()
    data[x_axis]=pd.to_numeric(data[x_axis],errors="coerce")
    data[y_axis]=pd.to_numeric(data[y_axis],errors="coerce")
    data=data.dropna(subset=[x_axis,y_axis])
    data=data.sort_values(x_axis)
    x=data[x_axis].to_numpy()
    y=data[y_axis].to_numpy()
    st.text(len(data))
    coeffs=np.polyfit(x,y,deg=12)
    poly_eq=np.poly1d(coeffs)
    y_fit=poly_eq(x)
    fig=pg.Figure()
    fig.add_trace(pg.Scatter(x=x,y=y,
                             mode="markers",
                             name="data"))
    fig.add_trace(pg.Scatter(x=x,y=y_fit,
                             mode="lines",
                             name="best fit 2nd degree"))
    
    fig.update_layout(xaxis_title={"text":x_axis,
                                   "font":{
                                       "size":20
                                    }
                                   },
                      yaxis_title={"text":y_axis,
                                   "font":{
                                       "size":20
                                    }
                                   },
                      title="Options plot")
    st.plotly_chart(fig)

global fig
fig=pg.Figure()

st.title("Option")
st.text("Welcome lets start trading")
st.sidebar.title("hello")
uploadfile=st.sidebar.file_uploader("Upload you file here")

if uploadfile:
    df=pd.read_csv(uploadfile)
    #Interactive(df)
col1,col2=st.columns(2)
with col1:
    st.header("X axis")
    option1=st.checkbox("strike x")
    options2=st.checkbox("IVs x")
    options3=st.checkbox("Model x")
    options4=st.checkbox("actual x")
    x=""
    y=""
    if option1:
        x="STRIKE"
    if options2:
        x="IV"
    if options3:
        x="Model_price"
    if options4:
        x="Actual_price"
with col2:
    st.header("Y axis")
    option1=st.checkbox("strike y")
    options2=st.checkbox("IVs y")
    options3=st.checkbox("Model y")
    options4=st.checkbox("actual y")
    if option1:
        y="STRIKE"
    if options2:
        y="IV"
    if options3:
        y="Model_price"
    if options4:
        y="Actual_price"
graph(x,y,df)





