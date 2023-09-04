import os
import tiktoken
import json

os.environ["OPENAI_API_KEY"] = "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"

urls = [
    'https://www.turing.com/blog/dotnet-architecture-framework-complete-guide/',
    'https://www.turing.com/blog/data-driven-it-optimization-guide/',
    'https://www.turing.com/blog/it-transformation-strategy-steps-best-practices/'
]

from langchain.document_loaders import UnstructuredURLLoader
loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)
docs = text_splitter.split_documents(data)

no_docs = len(docs)
print("No of Docs created", no_docs)
#print(docs)

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

import pickle
#vectorStore_openAI = FAISS.from_documents(docs, embeddings)
#with open("D:\python projects\Ahref scraping\store_openai.pkl", "wb") as f:
#    pickle.dump(vectorStore_openAI, f)


# Specify the full path to the pickle file
file_path = r"D:\python projects\Ahref scraping\store_openai.pkl"

with open(file_path, "rb") as f:
    VectorStore = pickle.load(f)

from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo')
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

Ans = chain({"question": "give me the best content structure to rank hing on google for It transformation"}, return_only_outputs=True)
print(Ans)
