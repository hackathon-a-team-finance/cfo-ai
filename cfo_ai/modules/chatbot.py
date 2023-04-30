import streamlit as st
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseRetriever

from cfo_ai.prompts.qa_templates import CONDENSE_QUESTION_PROMPT, CSV_QA_PROMPT


class Chatbot:
    """
    Chatbot with conversational chat
    """

    def __init__(self, model_name, temperature, use_retrieval, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.use_retrieval = use_retrieval
        self.vectors = vectors

    def conversational_chat(self, query):
        """
        Start a conversational chat
        """
        chain = None
        answer = None
        if self.use_retrieval:
            chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(
                    model_name=self.model_name, temperature=self.temperature
                ),
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                qa_prompt=CSV_QA_PROMPT,
                retriever=self.vectors.as_retriever(),
            )
            result = chain(
                {"question": query, "chat_history": st.session_state["history"]}
            )
            st.session_state["history"].append((query, result["answer"]))
            answer = result["answer"]
        else:
            chain = ConversationChain(
                llm=ChatOpenAI(
                    model_name=self.model_name, temperature=self.temperature
                ),
            )
            result = chain({"input": query, "history": st.session_state["history"]})
            st.session_state["history"].append((query, result["response"]))
            answer = result["response"]

        return answer
