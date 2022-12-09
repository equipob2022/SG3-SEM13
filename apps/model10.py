import streamlit as st
#importamos librerias
import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas_datareader as datas


def app():
    #start = '2004-08-18'
    #end = '2022-01-20'
    start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    end = st.date_input('End' , value=pd.to_datetime('today'))
    
    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil' , 'DOGE-EUR')

    df = datas.DataReader(user_input, 'yahoo', start, end)
    # Describiendo los datos
    st.subheader('Datos del 2004 al 2022') 
    st.write(df)
    st.subheader('Descripción de la dataset') 
    st.write(df.describe())