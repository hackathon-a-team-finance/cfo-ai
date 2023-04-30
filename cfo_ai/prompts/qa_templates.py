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
