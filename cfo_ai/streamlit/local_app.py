import os
import re
import sys
from io import BytesIO, StringIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

from cfo_ai.modules.chat_history import ChatHistory
from cfo_ai.modules.chatbot import Chatbot
from cfo_ai.modules.chatbot_utils import ChatbotUtils
from cfo_ai.modules.doc_utils import DocUtils
from cfo_ai.modules.embeddings import Embeddings
from cfo_ai.modules.sidebar import Sidebar


def prompt_form(file_type):
    """
    Displays the prompt form
    """
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area(
            "Query:",
            placeholder="Ask me anything about the {}: ".format(file_type),
            key="input",
            label_visibility="collapsed",
        )
        submit_button = st.form_submit_button(label="Send")

        is_ready = submit_button and user_input
    return is_ready, user_input


def show_header():
    """
    Displays the header of the app
    """
    st.markdown(
        """
            <h1 style='text-align: center;'>Talk with mAI CFO!</h1>
            """,
        unsafe_allow_html=True,
    )


def chat(chat_history, uploaded_file, mode, file_type):
    """
    Chat and generate messages
    """
    if st.session_state["ready"]:
        response_container, prompt_container = (
            st.container(),
            st.container(),
        )

        with prompt_container:
            is_ready, user_input = prompt_form(file_type)
            chat_history.initialize(uploaded_file)
            if st.session_state["reset_chat"]:
                chat_history.reset(uploaded_file)
            if is_ready:
                chat_history.append("user", user_input)
                if mode == "Q&A with CSV":
                    output = st.session_state["csv_agent"].run(user_input)
                elif mode == "Q&A with PDF":
                    output = st.session_state["chatbot"].conversational_chat(user_input)
                else:
                    output = "Invalid mode"
                chat_history.append("assistant", output)
        chat_history.generate_messages(response_container)
    return


def main():
    """
    Main function
    """
    load_dotenv()
    st.set_page_config(page_title="mAI CFO", page_icon=":book:")
    show_header()

    mode_options = ["Q&A with CSV", "Q&A with PDF", "Other"]

    sidebar = Sidebar()

    mode = st.sidebar.selectbox(
        "Select an option",
        mode_options,
    )

    st.session_state["mode"] = mode
    st.session_state["model"] = "gpt-3.5-turbo"
    st.session_state["temperature"] = 0.2

    model = st.session_state["model"]
    temperature = st.session_state["temperature"]

    if mode == "Q&A with CSV":
        doc_utils = DocUtils()
        sidebar.show_options()
        uploaded_file = doc_utils.handle_upload(file_type="csv")

        if uploaded_file:
            chat_history = ChatHistory(topic="transactions")
            uploaded_file_content = BytesIO(uploaded_file.getvalue())
            csv_agent = create_csv_agent(
                ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
                uploaded_file_content,
                verbose=True,
                max_iterations=4,
            )
            st.session_state["csv_agent"] = csv_agent
            st.session_state["ready"] = True

            try:
                chat(chat_history, uploaded_file, mode, file_type="csv")
            except Exception as e:
                st.error("Error: {}".format(e))
    elif mode == "Q&A with PDF":
        doc_utils = DocUtils(file_type="pdf")
        sidebar.show_options()
        uploaded_file = doc_utils.handle_upload(file_type="pdf")

        if uploaded_file:
            chat_history = ChatHistory(topic="mortgages")
            chatbot = ChatbotUtils.setup_chatbot(
                uploaded_file=uploaded_file,
                file_type="pdf",
                model=model,
                temperature=temperature,
                use_retrieval=True,
            )
            st.session_state["chatbot"] = chatbot

            try:
                chat(chat_history, uploaded_file, mode, file_type="pdf")
            except Exception as e:
                st.error("Error: {}".format(e))
    elif mode == "Other":
        print("In mode: Other")


if __name__ == "__main__":
    main()
