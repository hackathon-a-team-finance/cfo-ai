import os

import pandas as pd
import streamlit as st
from pypdf import PdfReader


class DocUtils:
    def __init__(self, file_type="csv"):
        self.file_type = file_type

    def show_user_file(self, uploaded_file):
        """Show user file"""
        file_container = st.expander("Your file :")
        if self.file_type == "csv":
            shows = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            file_container.write(shows)
        elif self.file_type == "pdf":
            uploaded_file.seek(0)
            file = uploaded_file.read()
        return

    def handle_upload(self, file_type):
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader(
            "upload", type=self.file_type, label_visibility="collapsed"
        )
        if uploaded_file is not None:
            self.show_user_file(uploaded_file)
        else:
            st.sidebar.info(
                "Upload your {} file to get started! ".format(file_type.upper())
            )
            st.session_state["reset_chat"] = True
        return uploaded_file
