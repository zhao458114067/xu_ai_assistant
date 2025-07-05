import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from pydantic import SecretStr
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from generate_vector_store import VECTOR_STORE_PATH

load_dotenv()
api_key = os.environ.get("API_KEY")

print("正在加载向量数据库...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH, embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=api_key,
    temperature=0.1,
    streaming=True  # 开启流式模式
)

print("AI 助手已启动，输入你的问题（输入 exit 退出）：\n")

history = []

while True:
    try:
        query = input("你问：")
        if query.strip().lower() in ["exit", "quit", "q"]:
            break

        docs = retriever.invoke(query)

        # 拼接上下文文档
        context = "\n\n".join([doc.page_content for doc in docs])

        # 构造历史对话,最多保留最近 10 轮
        history_prompt = ""
        for i, (q, a) in enumerate(history[-10:]):
            history_prompt += f"用户：{q}\n助手：{a}\n"

        # 构造完整 Prompt
        full_prompt = f"""你是一个知识问答助手，请根据以下文档内容和上下文对话回答用户的问题。
                        已知文档：
                        {context}
                
                        历史对话：
                        {history_prompt}
                
                        当前问题：
                        用户：{query}
                        助手："""

        # 输出答案（流式打印）
        print("回答：", end="", flush=True)
        answer = ""
        for chunk in llm.stream(full_prompt):
            print(chunk.content, end="", flush=True)
            answer += chunk.content
        print("\n")

        # 存入历史
        history.append((query, answer))

    except KeyboardInterrupt:
        break
