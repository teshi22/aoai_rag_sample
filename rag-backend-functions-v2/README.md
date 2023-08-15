curl -X POST -H "Content-Type: application/json" -d '@input_ask.json' http://localhost:7071/api/ask
curl -X POST -H "Content-Type: application/json" -d '@input_chat.json' http://localhost:7071/api/chat
curl -X POST -H "Content-Type: application/json" -d '@input.json' https://langchain-functions.azurewebsites.net/api/HttpTrigger1?code=s_WaChoNTpLmds0YN2JWHzg5Kcn-7oC153PzjVXy3f56AzFutkxx2Q==

"OPENAI_API_TYPE": "azure",
"OPENAI_API_KEY": "a6827ec52aa3419885dd433e5d68ec14",
"OPENAI_API_BASE": "https://openai-handson.openai.azure.com",
"OPENAI_DEPLOYMENT": "test-gpt-35-0613",
"AZURE_COGNITIVE_SEARCH_SERVICE_NAME": "cg-instance-115",
"AZURE_COGNITIVE_SEARCH_INDEX_NAME": "openai-doc-ja",
"AZURE_COGNITIVE_SEARCH_API_KEY": "fWvmrSAmFpmLQCrs7qJPe1DuRy6ywiBuDdgmcBC1QgAzSeCugaJC"

[
{
"name": "OPENAI_API_TYPE",
"value": "azure",
"slotSetting": false
},
{
"name": "OPENAI_API_KEY",
"value": "a6827ec52aa3419885dd433e5d68ec14",
"slotSetting": false
},
{
"name": "OPENAI_API_BASE",
"value": "https://openai-handson.openai.azure.com",
"slotSetting": false
},
{
"name": "OPENAI_DEPLOYMENT",
"value": "test-gpt-35-0613",
"slotSetting": false
},
{
"name": "AZURE_COGNITIVE_SEARCH_SERVICE_NAME",
"value": "cg-instance-115",
"slotSetting": false
},
{
"name": "AZURE_COGNITIVE_SEARCH_INDEX_NAME",
"value": "openai-doc-ja",
"slotSetting": false
},
{
"name": "AZURE_COGNITIVE_SEARCH_API_KEY",
"value": "fWvmrSAmFpmLQCrs7qJPe1DuRy6ywiBuDdgmcBC1QgAzSeCugaJC",
"slotSetting": false
}
]
