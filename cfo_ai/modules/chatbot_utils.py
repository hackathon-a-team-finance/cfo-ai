import os

import pandas as pd
import streamlit as st

from cfo_ai.modules.chatbot import Chatbot
from cfo_ai.modules.embeddings import Embeddings


class ChatbotUtils:
    @staticmethod
    def load_api_key():
        """
        Loads the OpenAI API key from the .env file or from the user's input
        and returns it
        """
        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
        else:
            user_api_key = st.sidebar.text_input(
                label="#### Your OpenAI API key 👇",
                placeholder="Paste your openAI API key, sk-",
                type="password",
            )
            if user_api_key:
                st.sidebar.success("API key l   oaded", icon="🚀")
        return user_api_key

    @staticmethod
    def setup_chatbot(uploaded_file, file_type, model, temperature, use_retrieval):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        embeddings = Embeddings()
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            vectors = embeddings.getDocEmbeds(file, uploaded_file.name, file_type)
            chatbot = Chatbot(
                model, temperature, use_retrieval=use_retrieval, vectors=vectors
            )
        st.session_state["ready"] = True
        return chatbot
