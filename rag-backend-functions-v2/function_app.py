import azure.functions as func
import logging
import os
import json
import requests

from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate


def function_get_index(query, llm):
    functions = [
        {
            "name": "search_doc",
            "description": "ドキュメント検索を行う",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索キーワード、日本語(例: Azure OpenAI, GPT, モデル)",
                    },
                    "index": {
                        "type": "string",
                        "enum": ["openai-doc-ja", "index-jinji", "other"],
                        "description": """検索の対象にするリソースの種類
                                            - 人事: index-jinji
                                            - OpenAI: openai-doc-ja
                                            - その他: other
                                        """,
                    },
                },
                "required": ["query", "index"],
            },
        }
    ]

    response = llm.predict_messages(
        [HumanMessage(content="Azure OpenAIで使えるモデルを教えて")], functions=functions
    )

    arg = json.loads(response.additional_kwargs["function_call"]["arguments"])
    print("#### Function Call Arguments ####")
    print(response)

    return arg["index"]


def get_prompt_template(prompt_num):
    prompt_list = []
    prompt_list.append(
        """
        ## 命令
        以下のコンテキストに基づいて、質問に対する回答を正確に生成してください。
        ## 制約
        - コンテキスト以外からは回答しないでください。
        - 質問で聞かれた内容以外を回答しないでください。
        - 回答に改行は使用せず、文章で回答してください。
        ## コンテキスト
        {context}
        ## 質問
        {question}
        ## 回答
        """
    )
    prompt_list.append(
        """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer in Japanese:"""
    )

    return prompt_list[prompt_num]


def qa_cognitive_search(query, index_name) -> dict:
    openai_base = os.getenv("OPENAI_API_BASE", "")
    openai_deployment = os.getenv("OPENAI_DEPLOYMENT", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    model_name = "gpt-35-turbo"

    search_key = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY", "")
    search_service_name = os.getenv("AZURE_COGNITIVE_SEARCH_SERVICE_NAME", "")

    llm = AzureChatOpenAI(
        model=model_name,
        deployment_name=openai_deployment,
        openai_api_base=openai_base,
        openai_api_key=openai_key,
        openai_api_version="2023-07-01-preview",
        temperature=0,
    )

    retriever = AzureCognitiveSearchRetriever(
        service_name=search_service_name,
        api_key=search_key,
        index_name=index_name,
        content_key="content",
        top_k=3,
    )

    PROMPT = PromptTemplate(
        template=get_prompt_template(0), input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    from langchain.callbacks import get_openai_callback

    with get_openai_callback() as cb:
        result = qa({"query": query})
        usage = {
            "completion_tokens": cb.completion_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "total_tokens": cb.total_tokens,
        }
        result["usage"] = usage
    result["model"] = model_name

    result["source_documents"] = [
        document_to_dict(doc) for doc in result["source_documents"]
    ]

    return result


def document_to_dict(doc):
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def chat(body):
    openai_base = os.getenv("OPENAI_API_BASE", "")
    deployment = os.getenv("OPENAI_DEPLOYMENT", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    url = "{}/openai/deployments/{}/chat/completions?api-version=2023-03-15-preview".format(
        openai_base, deployment
    )

    response = requests.post(
        url=url,
        headers={"Content-Type": "application/json", "api-key": api_key},
        data=json.dumps(body["request"]),
    )

    return json.loads(response.text)


app = func.FunctionApp()


@app.function_name(name="AskDocument")
@app.route(route="ask")
def function_ask(req: func.HttpRequest) -> func.HttpResponse:
    request_body = req.get_json()

    index = request_body["request"]["index"]
    query = request_body["request"]["query"]

    response = qa_cognitive_search(index_name=index, query=query)

    logging.info(
        json.dumps(
            {
                "user": request_body["user_name"],
                "request": request_body["request"],
                "response": response,
            },
            ensure_ascii=False,
        )
    )

    return func.HttpResponse(json.dumps(response, ensure_ascii=False))


@app.function_name(name="ChatGPT")
@app.route(route="chat")
def function_chat(req: func.HttpRequest) -> func.HttpResponse:
    request_body = req.get_json()
    response = chat(request_body)
    logging.info(
        json.dumps(
            {
                "user": request_body["user_name"],
                "request": request_body["request"],
                "response": response,
            },
            ensure_ascii=False,
        )
    )

    return func.HttpResponse(json.dumps(response, ensure_ascii=False))
