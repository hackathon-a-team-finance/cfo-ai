import streamlit as st
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseRetriever

from cfo_ai.prompts import qa_templates


class Chatbot:
    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow-up entry: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa_template = """"You are an AI conversational assistant to answer questions based on a context.
    You are given data from a csv file and a question, you must help the user find the information they need. 
    Your answers should be friendly, in the same language.
    question: {question}
    =========
    context: {context}
    =======
    """

    QA_PROMPT = PromptTemplate(
        template=qa_template, input_variables=["question", "context"]
    )

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
                condense_question_prompt=self.CONDENSE_QUESTION_PROMPT,
                qa_prompt=self.QA_PROMPT,
                retriever=BaseRetriever(),  # self.vectors.as_retriever() if self.vectors else None,
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

        print("In conversational_chat(), this is result: ", result)
        return answer
