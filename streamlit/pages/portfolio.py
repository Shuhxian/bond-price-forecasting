import streamlit as st
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def app():

    st.title('Portfolio')
    # Power BI report embedding
    powerbi_url= '<iframe width="1200" height="675" src="https://app.powerbi.com/reportEmbed?reportId=1e40c81c-a1d5-42fb-a5f5-cfb3907f65f3&autoAuth=true&ctid=a63bb1a9-48c2-448b-8693-3317b00ca7fb&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLXNvdXRoLWVhc3QtYXNpYS1yZWRpcmVjdC5hbmFseXNpcy53aW5kb3dzLm5ldC8ifQ%3D%3D" frameborder="0" allowFullScreen="true"></iframe>'
    # powerbi_url= '<iframe width="1200" height="675" src="https://app.powerbi.com/reportEmbed?reportId=665bb88f-abf0-480b-9217-78dcc6c4f024&autoAuth=true&ctid=a63bb1a9-48c2-448b-8693-3317b00ca7fb&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLXNvdXRoLWVhc3QtYXNpYS1yZWRpcmVjdC5hbmFseXNpcy53aW5kb3dzLm5ldC8ifQ%3D%3D" frameborder="0" allowFullScreen="true"></iframe>'
    st.markdown(powerbi_url,unsafe_allow_html=True)

    # Chart of bonds ordered by ratio in descending order
    # df = pd.DataFrame({
    # 'Bonds': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
    # 'Ratio': [2, 55, 43, 91, 81, 53, 19, 87, 52]
    # })
    # df = df.sort_values(by=['Ratio'], ascending=False).reset_index(drop=True)
    # fig, ax = plt.subplots()
    # ax.bar(df['Bonds'],df['Ratio'])
    # st.pyplot(fig, use_container_width=True)

    # Table of top 5 bonds
    # st.table(df.head(n=9))
    
    # Table of bond, return, volatility, and ratio