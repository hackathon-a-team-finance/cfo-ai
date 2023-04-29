import os
import re
import sys
from io import BytesIO, StringIO

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI


def init():
    print("Running init()..")


def main():
    init()
    print("hello world..")
    st.title("Sample app")


if __name__ == "__main__":
    main()
