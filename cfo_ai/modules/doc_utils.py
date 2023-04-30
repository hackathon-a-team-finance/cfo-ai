import os

import pandas as pd
import streamlit as st


class DocUtils:
    def __init__(self, file_type="csv"):
        self.file_type = file_type

    def show_user_file(self, uploaded_file):
        file_container = st.expander("Your CSV file :")
        shows = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(shows)
        return

    def handle_upload(self):
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader(
            "upload", type=self.file_type, label_visibility="collapsed"
        )
        if uploaded_file is not None:
            self.show_user_file(uploaded_file)
        else:
            st.sidebar.info("Upload your CSV file to get started! ")
            st.session_state["reset_chat"] = True
        return uploaded_file
