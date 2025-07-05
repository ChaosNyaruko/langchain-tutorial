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
llm_mistral = Ollama(model="mistral")
llm_llama31 = Ollama(model="llama3.1")
llm_llama3 = Ollama(model="llama3")
llm_llama2 = Ollama(model="llama2")
llm_phi4 = Ollama(model="phi4")
llm_qwen = Ollama(model="qwen:14b", temperature=0.1)
llm_ds = Ollama(model="deepseek-r1:8b")
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
prompt = PromptTemplate.from_template("""You are a language expert. Now I have some translations tasks. If the original text is given in Chinese, then translate into English, otherwise translate into Chinese. Please make your result idiomatic. The original text is "{question}" """)


from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
trivial_chain = prompt | llm_phi4 | output_parser
chat_chain = chat_prompt | llm_mistral | output_parser

qa_prompt_no_make_up = PromptTemplate.from_template("""{question}. If you don't know, just tell me that, DO NOT try to make it up""")

qa_prompt_plain = PromptTemplate.from_template("""{question}""")

qa_chain = qa_prompt_plain | llm_llama2 | output_parser

qa_chain_llama2 = qa_prompt_plain | llm_llama2 | output_parser
qa_chain_llama3 = qa_prompt_plain | llm_llama3 | output_parser
qa_chain_llama3_nomakeup = qa_prompt_no_make_up | llm_llama3 | output_parser
qa_chain_qwen = qa_prompt_plain | llm_qwen | output_parser
# ans = trivial_chain.invoke({"input": "There is no way (short of OCR) to extract text from these files."})
# print(ans + "\n---")
# ans = trivial_chain.invoke({"input": "What is your glorious purpose?"})
# print(ans + "\n---")
# ans = trivial_chain.invoke({"input": """请你给我讲一个笑话"""})
# print(ans + "\n---")

prompt_ch = PromptTemplate.from_template("""你是一个人类语言学专家。现在有一个翻译任务，如果原始文本是中文的，请把它翻译成英文，如果是其他的，都翻译成中文。最好能以信达雅的方式进行翻译，但不要曲解原义。你将要翻译的文本是: 
>>>
{question}
<<<
""")
ds_chain = prompt_ch | llm_ds | output_parser
qwen_chain = prompt_ch | llm_qwen | output_parser
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
add_routes(
    app,
    qa_chain_llama2,
    path="/qallama2",
)
add_routes(
    app,
    qa_chain_llama3,
    path="/qallama3",
)
add_routes(
    app,
    qa_chain_llama3_nomakeup,
    path="/qallama3_nomakeup",
)
add_routes(
    app,
    qa_chain_qwen,
    path="/qaqwen",
)
add_routes(
    app,
    qa_chain,
    path="/qa",
)
add_routes(
    app,
    ds_chain,
    path="/ds",
)
add_routes(
    app,
    qwen_chain,
    path="/qwen",
)

if __name__ == "__main__":
    # print(StrOutputParser().input_schema().schema())
    # print(StrOutputParser().output_schema().schema())
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
