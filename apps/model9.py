import streamlit as st


import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as datas
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import LSTM, Dense


def app():

    start = st.date_input('Inicio(Start)',value=pd.to_datetime('2010-01-01'))
    end = st.date_input('Fin' , value=pd.to_datetime('today'))
    
    st.title('Predicción de tendencia de acciones usando LSTM')

    user_input = st.text_input('Introducir cotización bursátil' , 'DOGE-EUR')
    startStr = start.strftime('%Y-%m-%d')
    endStr = end.strftime('%Y-%m-%d')
    dfi = datas.DataReader(user_input, 'yahoo', start, end)
    #renombra la columna de datetimne como Date
    dfi = dfi.reset_index()
    dfi = dfi.rename(columns={'Date':'Date'})
    #escribir un poco acerca de la empresa introducida en user_input
    # con la libreria de pandas_datareader podemos obtener informacion de la empresa
    st.subheader('Acerca de la empresa')
    st.write(datas.get_quote_yahoo(user_input))

    #hacemos un grafico de la serie de tiempo
    st.subheader('Serie de tiempo de la cotización bursátil de '+user_input)
    fig = px.line(dfi, x="Date", y="Close", title='Precio de cierre de '+ user_input)
    st.plotly_chart(fig)

    df = datas.DataReader(user_input, 'yahoo', start, '2021-12-31')
    df = df.reset_index()
    df = df.rename(columns={'Date':'Date'})

    test_df = datas.DataReader(user_input, 'yahoo', start, '2021-12-31')
    test_df = test_df.reset_index()
    test_df = test_df.rename(columns={'Date':'Date'})

    # ordenar por fecha
    df = df.sort_values('Date')
    test_df = test_df.sort_values('Date')

    # fix the date 
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    test_df.reset_index(inplace=True)
    test_df.set_index("Date", inplace=True)


    # cambiar las fechas en enteros para el entrenamiento
    dates_df = df.copy()
    dates_df = dates_df.reset_index()

    # Almacene las fechas originales para trazar las predicciones
    org_dates = dates_df['Date']

    # convert to ints
    dates_df['Date'] = dates_df['Date'].map(mdates.date2num)

    # Crear un conjunto de datos de entrenamiento de precios de 'Adj Close':
    train_data = df.loc[:,'Adj Close'].to_numpy()
    print(train_data.shape) # 1257 


    # Aplique la normalización antes de alimentar a LSTM usando sklearn:

    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)

    scaler.fit(train_data)
    train_data = scaler.transform(train_data)


    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    
    
    # Cree los datos para entrenar nuestro modelo en:
    time_steps = 36
    X_train, y_train = create_dataset(train_data, time_steps)

    # remodelarlo [muestras, pasos de tiempo, características]
    X_train = np.reshape(X_train, (X_train.shape[0], 36, 1))

    # Construye el modelo
    model = keras.Sequential()

    model.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100))
    model.add(Dropout(0.2))

    # Capa de salida
    model.add(Dense(units = 1))

    # Compilando el modelo
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Ajuste del modelo al conjunto de entrenamiento
    history = model.fit(X_train, y_train, epochs = 20, batch_size = 10, validation_split=.30)



    # Obtenga los precios de las acciones para 2019 para que nuestro modelo haga las predicciones
    test_data = test_df['Adj Close'].values
    test_data = test_data.reshape(-1,1)
    test_data = scaler.transform(test_data)

    # Cree los datos para probar nuestro modelo en:
    time_steps = 36
    X_test, y_test = create_dataset(test_data, time_steps)

    # almacenar los valores originales para trazar las predicciones
    y_test = y_test.reshape(-1,1)
    org_y = scaler.inverse_transform(y_test)

    # remodelarlo [muestras, pasos de tiempo, características]
    X_test = np.reshape(X_test, (X_test.shape[0], 36, 1))

    # Predecir los precios con el modelo.
    predicted_y = model.predict(X_test)
    predicted_y = scaler.inverse_transform(predicted_y)








    

    



   










