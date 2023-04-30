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


def init():
    print("Running init()..")


def prompt_form():
    """
    Displays the prompt form
    """
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area(
            "Query:",
            placeholder="Ask me anything about the document: ",
            key="input",
            label_visibility="collapsed",
        )
        submit_button = st.form_submit_button(label="Send")

        is_ready = submit_button and user_input
    return is_ready, user_input


def show_header(self):
    """
    Displays the header of the app
    """
    st.markdown(
        """
            <h1 style='text-align: center;'>Talk with mAI CFO!</h1>
            """,
        unsafe_allow_html=True,
    )


def main():
    """Main function"""
    init()
    load_dotenv()
    st.set_page_config(page_title="mAI CFO", page_icon=":book:")

    mode_options = ["Q&A with CSV", "Other"]

    sidebar = Sidebar()
    doc_utils = DocUtils()

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
        sidebar.show_options()
        uploaded_file = doc_utils.handle_upload()

        if uploaded_file:
            chat_history = ChatHistory(topic="transactions")
            uploaded_file_content = BytesIO(uploaded_file.getvalue())
            csv_agent = create_csv_agent(
                ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
                uploaded_file_content,
                verbose=True,
                max_iterations=4,
            )

            try:
                chatbot = ChatbotUtils.setup_chatbot(
                    uploaded_file_content, model, temperature, use_retrieval=False
                )
                st.session_state["chatbot"] = chatbot

                if st.session_state["ready"]:
                    response_container, prompt_container = (
                        st.container(),
                        st.container(),
                    )

                    with prompt_container:
                        is_ready, user_input = prompt_form()

                        chat_history.initialize(uploaded_file)
                        if st.session_state["reset_chat"]:
                            chat_history.reset(uploaded_file)

                        if is_ready:
                            chat_history.append("user", user_input)

                            if mode == "Q&A with CSV":
                                output = csv_agent.run(user_input)
                            elif mode == "Regular Q&A":
                                output = st.session_state[
                                    "chatbot"
                                ].conversational_chat(user_input)
                            else:
                                output = "Invalid mode"
                            chat_history.append("assistant", output)

                    chat_history.generate_messages(response_container)

            except Exception as e:
                st.error("Error: {}".format(e))
    elif mode == "Other":
        print("In mode: Other")


if __name__ == "__main__":
    main()
