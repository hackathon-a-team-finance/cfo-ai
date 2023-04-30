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
from cfo_ai.prompts.qa_templates import CSV_QA_PREFIX, CSV_QA_SUFFIX
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import PromptLayerChatOpenAI
from contextlib import contextmanager, redirect_stdout

def ask(input: str, agent: any) -> str:
    print("-- Serving request for input: %s" % input)
    try:
        response = agent.run(input)
    except Exception as e:
        response = str(e)
        if response.startswith("Could not parse LLM output: `"): #`
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    print("final response: ", response)
    return response

@contextmanager
def st_capture(output_function):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_function(escape_ansi(stdout.getvalue()))
            return ret
        
        stdout.write = new_write
        yield

def escape_ansi(line):
    return re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]').sub('', line)


def prompt_form(file_type):
    """
    Displays the prompt form
    """
    placeholder = "Ask me anything about the {}: ".format(file_type)
    if file_type == "none":
        placeholder = "Ask me anything about your finances!"
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area(
            "Query:",
            placeholder=placeholder,
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

def user_query_switch_csv(user_input):
    if "generate a P&L statement" in user_input or "Which month" in user_input:
        st.session_state["mode"] = "Analysis"

        csv_agent = create_csv_agent(
            ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
            path="cfo_ai/data/txn_data_sample.csv",
            verbose=True,
            max_iterations=4,
        )
        st.session_state["csv_agent"] = csv_agent
        st.session_state["ready"] = True
    return 

def chat(chat_history, uploaded_file, mode, file_type):
    """
    Chat and generate messages
    """
    if st.session_state["ready"]:
        response_container, prompt_container = (
            st.container(),
            st.container(),
        )
        print("this is mode session state var ****************: " , st.session_state["mode"])
        output = st.empty()

        with prompt_container:
            is_ready, user_input = prompt_form(file_type)
            print("this is session state mode BEFORE: ", st.session_state["mode"])
            #user_query_switch_csv(user_input)
            mode = st.session_state["mode"]
            print("this is session state mode AFTER: ", st.session_state["mode"])
            chat_history.initialize(uploaded_file)
            if st.session_state["reset_chat"]:
                chat_history.reset(uploaded_file)
            if is_ready:
                chat_history.append("user", user_input)
                if mode == "Analysis":
                    user_input = user_input + " Don't forget to use the structure of Action, Action Input, etc."
                    output = ask(user_input, st.session_state["csv_agent"])
                elif mode == "Document Analysis":
                    output = st.session_state["chatbot"].conversational_chat(user_input)
                elif mode == "General Q&A":
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

    mode_options = ["General Q&A", "Analysis", "Document Analysis", "Data Insights"]

    sidebar = Sidebar()

    mode = st.sidebar.selectbox(
        "Select an option",
        mode_options,
    )

    st.session_state["mode"] = mode
    st.session_state["model"] = "gpt-4"
    st.session_state["temperature"] = 0.1

    st.session_state["assistant_mode"] = "q&a"

    # Other
    st.session_state["integrate_bank_data"] = True

    model = st.session_state["model"]
    temperature = st.session_state["temperature"]


    if mode == "Analysis":
        doc_utils = DocUtils()
        sidebar.show_options()
        uploaded_file = doc_utils.handle_upload(file_type="csv")

        if uploaded_file:
            chat_history = ChatHistory(topic="transactions", mode="Analysis")
            uploaded_file_content = BytesIO(uploaded_file.getvalue())
            
            llm = PromptLayerChatOpenAI(temperature=0,model_name='gpt-4')
            csv_agent = create_csv_agent(llm, "cfo_ai/data/txn_data_sample.csv", verbose=True)

            # csv_agent = create_csv_agent(
            #     ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
            #     uploaded_file_content,
            #     verbose=True,
            #     max_iterations=4,
            # )

            st.session_state["csv_agent"] = csv_agent
            st.session_state["ready"] = True

            try:
                #ask(user_input, st.session_state["csv_agent"])
                chat(chat_history, uploaded_file, mode, file_type="csv")
            except Exception as e:
                st.error("Error: {}".format(e))
    elif mode == "Document Analysis":
        doc_utils = DocUtils(file_type="pdf")
        sidebar.show_options()
        uploaded_file = doc_utils.handle_upload(file_type="pdf", mode="Document Analysis")

        if uploaded_file:
            chat_history = ChatHistory(topic="mortgages", mode="Document Analysis")
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
    elif mode == "General Q&A":
        sidebar.show_options()

        chat_history = ChatHistory(topic="general", mode="General Q&A")

        chatbot = Chatbot(model, temperature, use_retrieval=False, vectors=None)

        st.session_state["chatbot"] = chatbot
        st.session_state["ready"] = True

        uploaded_file = None

        try:
            chat(chat_history, uploaded_file, mode, file_type="none")
        except Exception as e:
            st.error("Error: {}".format(e))
    elif mode == "Data Insights":
        st.markdown(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Iframe Example</title>
        </head>
        <body>
            <iframe src="https://analytics.penguins-lab.com/" width="100%" height="800px" frameborder="0" allowfullscreen>
                <!-- Fallback content displayed if the browser does not support iframes -->
                <p>Your browser does not support iframes. Please visit the <a href="https://penguinslab.streamlit.app/%22%3EPenguins Lab Streamlit App</a> website directly.</p>
            </iframe>
        </body>
        </html>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
