from langchain import PromptTemplate

financial_qa_template = """
    Instructions:
            - You are an experienced CFO of a small company.
            - You are an expert at managing finances, accounting, and taxes
            - You have access to detailed financial data. Use this data to provide insightful answers to financial queries.
            - Begin! 

    Question: {input}
    {agent_scratchpad}
    """

FINANCIAL_QUESTION_ANSWER_TEMPLATE_PROMPT = PromptTemplate(
    template=financial_qa_template,
    input_variables=["input", "agent_scratchpad"],
)


condense_template = """
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow-up entry: {question}
    Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)


csv_qa_template = """
    You are an AI conversational assistant to answer questions based on a context.
    You are given data from a csv file and a question, you must help the user find the information they need. 
    Your answers should be friendly, in the same language.

    question: {question}
    =========
    context: {context}
    =======
    """

CSV_QA_PROMPT = PromptTemplate(
    template=csv_qa_template, input_variables=["question", "context"]
)
