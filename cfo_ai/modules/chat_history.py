import os

import streamlit as st
from streamlit_chat import message


class ChatHistory:
    """
    Chat history
    """

    def __init__(self, topic, mode):
        self.history = st.session_state.get("history", [])
        self.topic = topic
        self.mode = mode
        st.session_state["history"] = self.history

    def default_greeting(self):
        return "Hello!"

    def default_prompt(self):
        qa_doc_prompt = f"Hi there, welcome to mAI CFO. Ask me about anything related to {self.topic}!"
        general_prompt = """
        Hi there, welcome to mAI CFO! Ask me about anything related to your finances, including but not limited to: 
        - Your financial transactions and history 
        - Company financial documents 
        - PnL statements 
        """
        if self.mode == "General Q&A":
            return general_prompt
        else:
            return qa_doc_prompt

    def initialize_user_history(self):
        st.session_state["user"] = [self.default_greeting()]

    def initialize_assistant_history(self, uploaded_file):
        st.session_state["assistant"] = [self.default_prompt()]

    def initialize(self, uploaded_file):
        """Initialize history"""
        if "assistant" not in st.session_state:
            self.initialize_assistant_history(uploaded_file)
        if "user" not in st.session_state:
            self.initialize_user_history()

    def reset(self, uploaded_file):
        """Reset history"""
        st.session_state["history"] = []
        self.initialize_user_history()
        self.initialize_assistant_history(uploaded_file)
        st.session_state["reset_chat"] = False

    def append(self, mode, message):
        """Append message to history"""
        st.session_state[mode].append(message)

    def generate_messages(self, container):
        if st.session_state["assistant"]:
            with container:
                for i in range(len(st.session_state["assistant"])):
                    message(
                        st.session_state["user"][i],
                        is_user=True,
                        key=f"{i}_user",
                        avatar_style="shapes",
                    )
                    message(
                        st.session_state["assistant"][i],
                        key=str(i),
                        avatar_style="fun-emoji",
                    )

    def load(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                self.history = f.read().splitlines()

    def save(self):
        with open(self.history_file, "w") as f:
            f.write("\n".join(self.history))
