import streamlit as st
import numpy as np
from sklearn.svm import SVR 
#import matplotlib.pyplot as plt 
import pandas as pd 


def app():
    st.title('Model 1 - SVR')
    data = yf.download('NTDOY')
    descrip=data.describe()
    st.write(descrip)