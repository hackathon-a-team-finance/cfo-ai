import os

import pandas as pd
import streamlit as st

from cfo_ai.modules.chatbot import Chatbot
from cfo_ai.modules.embeddings import Embeddings


class ChatbotUtils:
    """
    Chatbot utils
    """

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
