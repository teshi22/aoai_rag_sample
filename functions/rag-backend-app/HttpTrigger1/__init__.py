import logging
import os
import azure.functions as func

from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

OPENAI_API_TYPE = "azure"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT")

AZURE_COGNITIVE_SEARCH_SERVICE_NAME = os.getenv("AZURE_COGNITIVE_SEARCH_SERVICE_NAME") 
AZURE_COGNITIVE_SEARCH_INDEX_NAME = os.getenv("AZURE_COGNITIVE_SEARCH_INDEX_NAME")
AZURE_COGNITIVE_SEARCH_API_KEY = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    request_body = req.get_json()

    query = request_body.get("query") 
    logging.info("query: " + str(query))

    retriever = AzureCognitiveSearchRetriever(content_key="content")

    llm = ChatOpenAI(model_name='gpt-35-turbo', model_kwargs={"deployment_id": OPENAI_DEPLOYMENT}, temperature=0)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = qa({"query": query}) 

    logging.info("result: " + str(result["result"]))
    return func.HttpResponse(result["result"])

if __name__ == "__main__":
    print("Hello World")