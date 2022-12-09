import streamlit as st
import talib
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas_datareader as datas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn import metrics

def app():
    st.title('Random Forest - Cambio de Dogecoin')

    start = st.date_input('Inicio',value=pd.to_datetime('2017-11-28'))
    end = st.date_input('Fin' , value=pd.to_datetime('today'))

    startStr = start.strftime('%Y-%m-%d')
    endStr = end.strftime('%Y-%m-%d')

    user_input = st.text_input('Introducir cotización bursátil' , 'DOGE-USD')
   
    df = datas.DataReader(user_input, 'yahoo', start, end)


    st.title('Predicción de tendencia de acciones')

    # Visualizaciones
    st.subheader('Precio de cierre ajustado')
    fig = px.line(df,y='Adj Close')
    st.plotly_chart(fig)

    st.subheader('Patron de cambio diario')
    fig = plt.figure(figsize = (12,6))
    df['Adj Close'].pct_change().plot.hist(bins=50)
    plt.xlabel("Adjusted close 1 day percent change")
    st.pyplot(fig)

    st.subheader('Feature engineering')

    #Creamos una lista vacia de caracteristicas
    feature_names = []
    #Calculo en loop de los MA y RSI en los periodos
    for n in [14, 30, 50, 200]:
        df['ma' + str(n)] = talib.SMA(df['Adj Close'].values, timeperiod=n)
        df['rsi' + str(n)] = talib.RSI(df['Adj Close'].values, timeperiod=n)
    
    #Agregando estas caracteristicas a la lista
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

    df['Volume_1d_change'] = df['Volume'].pct_change()

    volume_features = ['Volume_1d_change']
    feature_names.extend(volume_features)

    df['5d_future_close'] = df['Adj Close'].shift(-5)
    df['5d_close_future_pct'] = df['5d_future_close'].pct_change(5)

    df.dropna(inplace=True)

    st.write(df)

    X = df[feature_names]
    y = df['5d_close_future_pct']

    train_size = int(0.85 * y.shape[0])

    #Definiendo al conjunto de entrenamiento con el 85%
    X_train = X[:train_size]
    y_train = y[:train_size]
    #Definiendo al conjunto de entrenamiento con el 15% restante
    X_test = X[train_size:]
    y_test = y[train_size:]

    rf_model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=3, 
        max_features=8, \
        random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    st.subheader('Prediccion con Random Forest')

    y_pred_series = pd.Series(y_pred, index=y_test.index)
    fig = px.line(y_pred_series)
    st.plotly_chart(fig)

    ## Métricas
    MAE=metrics.mean_absolute_error(y_test, y_pred)
    MSE=metrics.mean_squared_error(y_test, y_pred)
    RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    metricas = {
        'metrica' : ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
        'valor': [MAE, MSE, RMSE]
    }
    metricas = pd.DataFrame(metricas)

    ### Gráfica de las métricas
    st.subheader('Métricas de rendimiento')
    
    st.write(pd.DataFrame(
        data = [MSE, MAE, RMSE],
        index = ["MSE", "MAE", "MAPE"],
        columns = ["Error"]
    ))

    fig = px.bar(        
        metricas,
        x = "metrica",
        y = "valor",
        title = "Métricas del Modelo Random Forest Regressor",
        color="metrica"
    )
    st.plotly_chart(fig)

    
    