import streamlit as st
from multiapp import MultiApp
from pages import dashboard, dataset_page, model_page, portfolio # import your app modules here

app = MultiApp()

st.title('GigaBITS')

# Add all your application here
app.add_app("Dashboard", dashboard.app)
app.add_app("Portfolio", portfolio.app)
app.add_app("Model", model_page.app)
app.add_app("Dataset", dataset_page.app)

# The main app
app.run()