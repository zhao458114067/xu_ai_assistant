import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from generate_vector_store import VECTOR_STORE_PATH

load_dotenv()
api_key = os.environ.get("API_KEY")

print("📚 正在加载向量数据库...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH, embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=SecretStr(api_key),
    temperature=0.7,
    streaming=True  # 开启流式模式
)

print("\n🤖 AI 助手已启动，输入你的问题（输入 exit 退出）：\n")

while True:
    try:
        query = input("你问：")
        if query.strip().lower() in ["exit", "quit", "q"]:
            break

        docs = retriever.invoke(query)

        # 拼接上下文（也可以更复杂地构造 Prompt）
        context = "\n\n".join([doc.page_content for doc in docs])
        full_prompt = f"基于以下内容回答问题：\n{context}\n\n问题：{query}\n回答："

        # ✅ 流式打印回答
        print("回答：", end="", flush=True)
        for chunk in llm.stream(full_prompt):
            print(chunk.content, end="", flush=True)
        print("\n")

    except KeyboardInterrupt:
        break
