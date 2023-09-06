import azure.functions as func
import logging
import os
import json
import requests

from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.callbacks import get_openai_callback
from langchain.utilities import BingSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory


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


@app.function_name(name="Agent")
@app.route(route="agent")
def function_agent(req: func.HttpRequest) -> func.HttpResponse:
    request_body = req.get_json()

    # Azure OpenAIのエンドポイントとAPIキー
    AZURE_OPENAI_ENDPOINT = os.getenv("OPENAI_API_BASE", "")
    AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    AZURE_OPENAI_MODEL_NAME = "gpt-3.5-turbo"
    AZURE_OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT", "")

    # Azure SearchのエンドポイントとAPIキー
    AZURE_SEARCH_SERVICE_NAME = os.getenv("AZURE_COGNITIVE_SEARCH_SERVICE_NAME", "")
    AZURE_SEARCH_API_KEY_ADMIN = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY", "")
    AZURE_COGNITIVE_SEARCH_INDEX_NAME = os.getenv(
        "AZURE_COGNITIVE_SEARCH_INDEX_NAME", ""
    )
    BING_SEARCH_URL = os.getenv("BING_SEARCH_URL", "")
    BING_SUBSCRIPTION_KEY = os.getenv("BING_SUBSCRIPTION_KEY", "")

    chat = AzureChatOpenAI(
        openai_api_type="azure",
        model=AZURE_OPENAI_MODEL_NAME,
        openai_api_version="2023-07-01-preview",
        openai_api_base=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        temperature=0,
    )

    retriever = AzureCognitiveSearchRetriever(
        service_name=AZURE_SEARCH_SERVICE_NAME,
        api_key=AZURE_SEARCH_API_KEY_ADMIN,
        index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME,
        content_key="content",
        top_k=3,
    )

    cognitive_tool = create_retriever_tool(
        retriever=retriever,
        name="serach_azure_openai_document",
        # description="useful for when you need to answer questions about Azure OpenAI",
        description="Azure OpenAI に関する質問に答える必要がある場合に役立ちます",
    )

    search = BingSearchAPIWrapper(
        bing_search_url=BING_SEARCH_URL,
        bing_subscription_key=BING_SUBSCRIPTION_KEY,
        k=3,
    )

    bing_tool = Tool.from_function(
        func=search.run,
        name="search_bing",
        description="useful for when you need to answer questions about current events",
    )

    useBing = False  # Bingを使う場合はTrueにする

    if useBing:
        tools = [cognitive_tool, bing_tool]
    else:
        tools = [cognitive_tool]

    system_message = SystemMessage(
        content="""
        質問への回答を生成してください。
        会話のやり取りはすべて日本語で行ってください。
        検索に利用できるツールがあれば自由に使用してください。
        必要な場合のみ関連情報を日本語で検索してください。
        serach_azure_openai_documentを使う場合は必ず質問文を変更せずに、日本語で検索してください。
        """
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    agent = OpenAIFunctionsAgent(llm=chat, tools=tools, prompt=prompt)

    memory = ConversationBufferWindowMemory(
        k=3, return_messages=True, memory_key="chat_history", output_key="output"
    )

    input_dict = {"input": ""}
    output_dict = {"output": ""}

    for message in request_body["request"]["messages"]:
        if message["role"] == "user":
            input_dict["input"] = message["content"]
        elif message["role"] == "assistant":
            output_dict["output"] = message["content"]
            memory.save_context(input_dict, output_dict)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )

    query = request_body["request"]["messages"][-1]["content"]

    with get_openai_callback() as cb:
        result = agent_executor({"input": query})
        usage = {
            "completion_tokens": cb.completion_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "total_tokens": cb.total_tokens,
        }
        result["usage"] = usage

    result["model"] = AZURE_OPENAI_MODEL_NAME

    # クラスを辞書に変換する関数
    def class_to_dict(obj):
        if isinstance(obj, list):
            return [class_to_dict(e) for e in obj]
        elif isinstance(obj, tuple):
            return tuple(class_to_dict(e) for e in obj)
        elif isinstance(obj, dict):
            return {k: class_to_dict(v) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            return {k: class_to_dict(v) for k, v in obj.__dict__.items()}
        else:
            return obj

    del result["chat_history"]
    result["intermediate_steps"] = class_to_dict(result["intermediate_steps"])

    logging.info(
        json.dumps(
            {
                "type": "output",
                "user": request_body["user_name"],
                "request": request_body["request"],
                "response": result,
            },
            ensure_ascii=False,
        )
    )

    return func.HttpResponse(json.dumps(result, ensure_ascii=False))


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
