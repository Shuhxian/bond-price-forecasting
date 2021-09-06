import streamlit as st
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def app():
    st.title('Portfolio')

    # Chart of bonds ordered by ratio in descending order
    df = pd.DataFrame({
    'Bonds': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
    'Ratio': [2, 55, 43, 91, 81, 53, 19, 87, 52]
    })
    df = df.sort_values(by=['Ratio'], ascending=False).reset_index(drop=True)
    # fig, ax = plt.subplots()
    # ax.bar(df['Bonds'],df['Ratio'])
    # st.pyplot(fig, use_container_width=True)

    # Table of top 5 bonds
    st.table(df.head(n=9))
    