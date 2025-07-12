# test_app.py - Minimal version for testing
import streamlit as st

st.title("SAP EWA Test")
st.write("If you see this, basic Streamlit works!")

# Test imports one by one
try:
    import pandas as pd
    st.success("✅ Pandas imported")
except Exception as e:
    st.error(f"❌ Pandas error: {e}")

try:
    import plotly.express as px
    st.success("✅ Plotly imported")
except Exception as e:
    st.error(f"❌ Plotly error: {e}")

try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    email = os.getenv("GMAIL_EMAIL")
    st.success(f"✅ Dotenv works, email: {email}")
except Exception as e:
    st.error(f"❌ Dotenv error: {e}")

st.write("Test complete!")