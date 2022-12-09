import streamlit as st
from matplotlib import style
import seaborn as sns
from scipy.stats import randint as sp_randint
from sklearn.decomposition import PCA
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import yfinance as yf
import pandas_datareader as datas
def app():    
    st.title('Modelo 12 - PCA and Hierarchical Portfolio Optimisation')
    start = st.date_input('Start' , value=pd.to_datetime('2018-01-01'))
    end = st.date_input('End' , value=pd.to_datetime('today'))

    ticker= "NTDOY "
    stock_data = yf.download(ticker, start="2018-01-01", end="2022-12-8")   

    user_input = st.text_input('Introducir cotización bursátil' , 'NTDOY',disabled=False)

    df = datas.DataReader(user_input, 'yahoo', start, end)
    st.subheader('Datos del 2018 al 2022') 
    st.write(df.describe())
    