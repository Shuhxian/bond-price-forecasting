from altair.vegalite.v4.schema.channels import X2Value
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

def app():
    st.title('Model')
    st.markdown('## Current Model')
    col1, col2 = st.columns(2)
    with col1:
        st.write('Accuracy')
        x = np.arange(1,100)
        source = pd.DataFrame({
        'x': x,
        'f(x)': np.log(x)*20
        })

        chart = alt.Chart(source).mark_line().encode(
            x='x',
            y='f(x)'
        )
        st.altair_chart(chart, use_container_width=True)
        
    with col2:
        st.write('Loss')
        x = np.arange(1,100)
        source = pd.DataFrame({
        'x': x,
        'f(x)': [1/i for i in x]
        })

        chart = alt.Chart(source).mark_line().encode(
            x='x',
            y='f(x)'
        )
        st.altair_chart(chart, use_container_width=True)

