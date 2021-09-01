import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

def main():
    # Sidebar
    st.sidebar.header('GigaBITS')

    dashboard = st.sidebar.button('Dashboard')

    portfolio = st.sidebar.button('Portfolio')

    model = st.sidebar.button('Model')

    dataset = st.sidebar.button('Dataset')

    # dashboard
    if dashboard:
        st.header('Dashboard')
        #

    # portfolio
    elif portfolio:
        st.header('Portfolio')

    elif model:
        st.header('Model')

    elif dataset:
        st.header('Dataset')



if __name__=='__main__':
    main()