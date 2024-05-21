from typing import List

from fastapi import FastAPI
from langserve import add_routes
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
# from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
llm = Ollama(model="llama3", temperature=0)
# prompt = ChatPromptTemplate.from_messages([
#     ("user", "{input}"),
#     ("user", "Given the above English text, translate them into Simplified Chinese please. NOTE: This sentence is NOT included.")
# ])

system_template ="""Please translate some text. If the original text is in English, translate info Simplified Chinese, otherwise, translate into English. The output should include:
1. The original text and its language.
2. The target text and its language.
If possible,  some cultural or contextual adjustments can be applied, to make it more idiomatic."""

chat_prompt= ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "我的文本 {text}")]
)
prompt = PromptTemplate.from_template("""Please translate some text. If the original text is in English, translate info Simplified Chinese, otherwise, translate into English. The output should include:
1. The original text and its language.
2. The target text and its language.
If possible,  some cultural or contextual adjustments can be applied, to make it more idiomatic.

ORIGINAL TEXT:
{input}
""")

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
trivial_chain = prompt | llm | output_parser
chat_chain = chat_prompt | llm | output_parser
# ans = trivial_chain.invoke({"input": "There is no way (short of OCR) to extract text from these files."})
# print(ans + "\n---")
# ans = trivial_chain.invoke({"input": "What is your glorious purpose?"})
# print(ans + "\n---")
# ans = trivial_chain.invoke({"input": """请你给我讲一个笑话"""})
# print(ans + "\n---")
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    trivial_chain,
    path="/chain",
)
add_routes(
    app,
    chat_chain,
    path="/chainv1",
)

if __name__ == "__main__":
    # print(StrOutputParser().input_schema().schema())
    # print(StrOutputParser().output_schema().schema())
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
