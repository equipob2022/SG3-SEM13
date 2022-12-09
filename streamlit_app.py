
import streamlit as st
from multiapp import MultiApp
# import los modelos aqui
from apps import model10 

app = MultiApp()

st.markdown("""
# Inteligencia de Negocios - Equipo B
""")
# Add all your application here
# app.add_app("Homea", home.app)
# app.add_app("Modelo SVR", model1.app)
# app.add_app("PCA and Hierarchical Portfolio Optimisation", model12.app)
app.add_app("Modelo XGBoost", model10.app)
# app.add_app("Modelo ARIMA", model4.app)

# The main app
app.run()