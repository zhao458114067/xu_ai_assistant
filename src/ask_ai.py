import os
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from generate_vector_store import VECTOR_STORE_PATH, model_name, load_documents, DATA_PATH
from src.embedding.onnx_embeddings import OnnxEmbeddings
from src.onnx_export import output_dir


def start_assistant():
    print("正在加载向量数据库...")
    embeddings = OnnxEmbeddings(
        onnx_path=os.path.join(output_dir, model_name.replace("/", "_") + ".onnx"),
        model_name=model_name
    )
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 检索其
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 15,
            "fetch_k": 30,
            "lambda_mult": 0.7
        }
    )

    documents = load_documents(DATA_PATH)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 15
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.7, 0.3]
    )

    # 大模型
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ.get("API_KEY"),
        temperature=0.3,
        streaming=True  # 开启流式模式
    )

    print("AI 助手已启动，输入你的问题（输入 exit 退出）：\n")
    question_and_answer(retriever, llm)


def question_and_answer(retriever, llm):
    history = []

    rewrite_sys_message = SystemMessage(
        content="请你将用户的问题提取出最有助于在代码或文档中进行检索的关键词列表。返回时只输出关键词，用空格隔开，不要解释或添加前缀。保留专业术语、函数名、变量名等关键词，不要转换为自然语言。")
    ask_sys_message = SystemMessage(content="你是一个熟悉代码和技术文档的助手。请根据以下资料回答用户问题。")
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

            rewrite_prompt = f"""
            【文档】\n
            {doc_contents}\n

            【用户问题】\n
            {query}\n
            """
            print("拆解提问中...\n")
            rewrite_prompt = get_llm_answer(llm,
                                            [rewrite_sys_message] + messages + [HumanMessage(content=rewrite_prompt)])

            doc_contents = retrieve_contents(rewrite_prompt, retriever)

            # 构造完整 Prompt
            full_prompt = f"""
            【文档】\n
            {doc_contents}\n

            【用户问题】\n
            {query}\n
            """

            messages.append(HumanMessage(content=full_prompt))

            # 输出答案（流式打印）
            print("回答：", end="", flush=True)
            answer = get_llm_answer(llm, [ask_sys_message] + messages)
            print("\n")

            # 存入历史
            history.append((query, answer))

        except KeyboardInterrupt:
            break


def get_llm_answer(llm, messages):
    answer = ""
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
        answer += chunk.content
    print("\n")
    return answer


def retrieve_contents(query, retriever):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


if __name__ == '__main__':
    load_dotenv()
    start_assistant()
