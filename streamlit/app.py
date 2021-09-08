import streamlit as st
from multiapp import MultiApp
from pages import dataset_page, model_page, portfolio, train_page # import your app modules here

app = MultiApp()

st.title('GigaBITS')

max_width = 1200
padding_top = 5
padding_left = 1
padding_right = 1
padding_bottom = 10

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
</style>
""",
        unsafe_allow_html=True,
)

# Add all your application here
app.add_app("Portfolio", portfolio.app)
app.add_app("Model", model_page.app)
app.add_app("Dataset", dataset_page.app)
app.add_app("Train New Model", train_page.app)

# The main app
app.run()