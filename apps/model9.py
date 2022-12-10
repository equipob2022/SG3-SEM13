# import streamlit as st


# import numpy as np
# np.random.seed(4)
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import pandas as pd
# import pandas_datareader as datas
# import plotly.express as px
# from sklearn.preprocessing import MinMaxScaler
# import keras
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.models import Sequential
# from keras.layers import LSTM, Dense


# def app():

#     start = st.date_input('Inicio(Start)',value=pd.to_datetime('2010-01-01'))
#     end = st.date_input('Fin' , value=pd.to_datetime('today'))
    
#     st.title('Predicción de tendencia de acciones usando LSTM')

#     user_input = st.text_input('Introducir cotización bursátil' , 'DOGE-EUR')
#     dfi = datas.DataReader(user_input, 'yahoo', start, end)
#     #renombra la columna de datetimne como Date
#     dfi = dfi.reset_index()
#     dfi = dfi.rename(columns={'Date':'Date'})
#     #escribir un poco acerca de la empresa introducida en user_input
#     # con la libreria de pandas_datareader podemos obtener informacion de la empresa
#     st.subheader('Acerca de la empresa')
#     st.write(datas.get_quote_yahoo(user_input))
#     dfi.reset_index(inplace=True)
#     dfi.set_index("Date", inplace=True)

#     #hacemos un grafico de la serie de tiempo
#     st.subheader('Serie de tiempo de la cotización bursátil de '+user_input)
#     fig = px.line(dfi, x="Date", y="Close", title='Precio de cierre de '+ user_input)
#     st.plotly_chart(fig)

