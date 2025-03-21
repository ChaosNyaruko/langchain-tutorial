from typing import List

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
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
prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}"),
    ("user", "Given the above English text, translate them into Simplified Chinese please. NOTE: This sentence is NOT included.")
])

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
trivial_chain = prompt | llm | output_parser
ans = trivial_chain.invoke({"input": "It is the punch line"})
print(ans)
