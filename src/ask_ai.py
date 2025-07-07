import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from generate_vector_store import VECTOR_STORE_PATH
from src.embedding.onnx_embeddings import OnnxEmbeddings
from src.onnx_export import output_dir


def start_assistant():
    print("正在加载向量数据库...")
    model_name = "intfloat/multilingual-e5-large"
    embeddings = OnnxEmbeddings(
        onnx_path=os.path.join(output_dir, model_name.replace("/", "_") + ".onnx"),
        model_name=model_name
    )
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH, embeddings,
        allow_dangerous_deserialization=True
    )

    # 检索其
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 20,
            "lambda_mult": 0.9
        }
    )

    # retriever = vectorstore.as_retriever(
    #     # search_type="similarity",
    #     search_kwargs={"k": 10}
    # )

    # 大模型
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ.get("API_KEY"),
        temperature=0.7,
        streaming=True  # 开启流式模式
    )

    print("AI 助手已启动，输入你的问题（输入 exit 退出）：\n")
    question_and_answer(retriever, llm)


def question_and_answer(retriever, llm):
    history = []

    while True:
        try:
            query = input("你问：")
            if query.strip().lower() in ["exit", "quit", "q"]:
                break
            # 检索文档
            doc_contents = retrieve_contents(query, retriever)

            # 构造历史对话,最多保留最近 5 轮
            messages = []
            for q, a in history[-5:]:
                messages.append(HumanMessage(content=q))
                messages.append(AIMessage(content=a))

            rewrite_prompt = (f"""请你基于上下文，把这个问题补充成更完整、更有助于langchain向量库检索的形式。注意：优化后的语句不需要任何解释或前缀。
                              \n\n用户问题：{query}
                                """)
            rewrite_prompt = llm.invoke(messages + [HumanMessage(content=rewrite_prompt)]).content.strip()

            doc_contents = retrieve_contents(rewrite_prompt, retriever)

            # 构造完整 Prompt
            full_prompt = f"""请根据以下文档回答问题。文档以中文为主，请勿编造。”。        
            \n\n资料：{doc_contents}            
            \n\n用户问题：{query}
            """

            messages.append(HumanMessage(content=full_prompt))

            # 输出答案（流式打印）
            print("回答：", end="", flush=True)
            answer = ""
            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)
                answer += chunk.content
            print("\n")

            # 存入历史
            history.append((query, answer))

        except KeyboardInterrupt:
            break


def retrieve_contents(query, retriever):
    docs = retriever.invoke(f"query: {query.strip()}")
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


if __name__ == '__main__':
    load_dotenv()
    start_assistant()
