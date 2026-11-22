from APP.agent import graph

result = graph.invoke(
    {
        "question": "What are common symptoms of anxiety?",
        "image_description": ""
    },
    config={
        "configurable": {
            "thread_id": "therapisto-test-001"
        }
    }
)

print(result["answer"])