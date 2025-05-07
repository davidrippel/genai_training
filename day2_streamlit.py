import streamlit as st
from streamlit.logger import get_logger
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

logger = get_logger(__name__)

import os
if os.getenv('USER', "None") == 'appuser': # streamlit
    hf_token = st.secrets['HF_TOKEN']
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
else:
    # ALSO ADD HERE YOUR PROXY VARS
    os.environ["HTTP_PROXY"] = "http://proxy-dmz.intel.com:912"
    os.environ["HTTPS_PROXY"] = "http://proxy-dmz.intel.com:912"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["MY_HF_API_TOKEN"]

# Title
st.title("My First Gen AI App")

# Repo ID and Temperature
repo_id_default = "microsoft/Phi-3-mini-4k-instruct"
temperature_default = 1.0
st.write("Enter the model repo id and temperature")
repo_id = st.text_input("Model repo id:", repo_id_default)
# temperature = st.number_input("Temperature:", 0.01, 2.01, temperature_default, 0.1)
temperature = st.slider("Temperature:", 0.1, 2.0, temperature_default, 0.1)
print(f"{repo_id=} {temperature=}")
logger.info(f"{repo_id=} {temperature=} ")

# Create the form
with st.form("Generate Text"):
    txt = st.text_area("Enter Text", "What is the capital of France?")
    sub = st.form_submit_button("submit")
    if sub:
        llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                task="text-generation",
                temperature=temperature
                )
        chat = ChatHuggingFace(llm=llm, verbose=True)
        logger.info("invoking")
        ans = chat.invoke(txt)
        st.info(ans.content)
        logger.info("Done")

