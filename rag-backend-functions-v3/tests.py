import unittest
import os
import json

import azure.functions as func
from function_app import (
    qa_cognitive_search,
    chat,
    function_chat,
    function_ask,
    function_agent,
)


class TestFunction(unittest.TestCase):
    def setUp(self):
        file_path = os.path.join(os.path.dirname(__file__), "local.settings.json")
        with open(file_path, "r") as f:
            env = json.load(f)["Values"]
        for env_key in env.keys():
            os.environ[env_key] = env[env_key]

    @unittest.skip("skip")
    def test_qa_cognitive_search(self):
        index = "openai-doc-ja"
        query = "Azure OpenAIで使えるモデルを教えて"

        result = qa_cognitive_search(index_name=index, query=query)

        print(json.dumps(result, indent=2, ensure_ascii=False))

    @unittest.skip("skip")
    def test_chat(self):
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps people find information.",
                },
                {"role": "user", "content": "こんにちわ"},
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 800,
            "stop": None,
        }

        result = chat(body)

        print(json.dumps(result, indent=2, ensure_ascii=False))

    @unittest.skip("skip")
    def test_agent(self):
        body = {
            "request": {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that helps people find information.",
                    },
                    {"role": "user", "content": "東京の首都は？"},
                    {"role": "assistant", "content": "東京は日本の首都です。"},
                    {"role": "user", "content": "アメリカは？"},
                ],
            },
            "user_name": "test",
        }

        req = func.HttpRequest(
            method="POST", body=json.dumps(body).encode("utf-8"), url="/api/agent"
        )

        func_call = function_agent.build().get_user_function()
        resp = func_call(req)
        result = json.loads(resp.get_body().decode("utf-8"))

        # print(json.dumps(result, indent=2, ensure_ascii=False))

    def test_agent_2(self):
        test_list = ["Azure OpenAIで使えるモデルを教えて", "アメリカは?", "2023年のWBC優勝国は？", "雨はなぜ降るの？"]
        body = {
            "request": {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that helps people find information.",
                    },
                    {"role": "user", "content": "東京の首都は？"},
                    {"role": "assistant", "content": "東京は日本の首都です。"},
                    {"role": "user", "content": test_list[3]},
                ],
            },
            "user_name": "test",
        }

        req = func.HttpRequest(
            method="POST", body=json.dumps(body).encode("utf-8"), url="/api/agent"
        )

        func_call = function_agent.build().get_user_function()
        resp = func_call(req)
        result = json.loads(resp.get_body().decode("utf-8"))

        # print(json.dumps(result, indent=2, ensure_ascii=False))

    @unittest.skip("skip")
    def test_function_chat(self):
        body = {
            "request": {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that helps people find information.",
                    },
                    {"role": "user", "content": "こんにちわ"},
                ],
                "temperature": 0.7,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "max_tokens": 800,
            },
            "user_name": "test",
        }

        req = func.HttpRequest(
            method="POST", body=json.dumps(body).encode("utf-8"), url="/api/chat"
        )

        func_call = function_chat.build().get_user_function()
        resp = func_call(req)
        result = json.loads(resp.get_body().decode("utf-8"))

        print(json.dumps(result, indent=2, ensure_ascii=False))

    @unittest.skip("skip")
    def test_function_ask(self):
        body = {
            "request": {
                "index": "openai-doc-ja",
                "query": "Azure OpenAIで使えるモデルを教えて",
            },
            "user_name": "test",
        }

        req = func.HttpRequest(
            method="POST", body=json.dumps(body).encode("utf-8"), url="/api/ask"
        )

        func_call = function_ask.build().get_user_function()
        resp = func_call(req)
        result = json.loads(resp.get_body().decode("utf-8"))

        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
