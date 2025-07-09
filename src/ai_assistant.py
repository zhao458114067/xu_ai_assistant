import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.embedding.onnx_embeddings import OnnxEmbeddings
from src.generate_vector_store import VECTOR_STORE_PATH, load_documents, DATA_PATH
from src.handler.websocket_call_back_handler import WebSocketCallbackHandler
from src.onnx_export import output_dir, model_name


class AIAssistantService:
    def __init__(self):
        load_dotenv()
        self.llm = None
        self.retriever = None
        self.user_sessions: Dict[str, Dict[str, Any]] = {}

        # 初始化向量数据库和检索器
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

        # 初始化检索器
        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 30,
                "lambda_mult": 0.7
            }
        )

        documents = load_documents(DATA_PATH)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.7, 0.3]
        )

        # 初始化大模型
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("API_KEY"),
            temperature=0.3,
            streaming=True
        )

        self.rewrite_sys_message = SystemMessage(
            content="请你将用户的问题提取出最有助于在代码或文档向量库中进行检索的关键词列表。返回时只输出关键词，用空格隔开，不要解释或添加前缀。保留专业术语、函数名、变量名等关键词，不要转换为自然语言。")
        self.ask_sys_message = SystemMessage(content="你是一个熟悉代码和技术文档的助手。请根据以下资料回答用户问题。")

    async def retrieve_contents(self, query):
        docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

    def get_user_session(self, user_id: str):
        """获取或创建用户会话"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'history': [],
                'task': None,
                'active': False
            }
        return self.user_sessions[user_id]

    async def process_query(self, user_id: str, query: str, websocket):
        user_session = self.get_user_session(user_id)

        # 第一次检索
        doc_contents = await self.retrieve_contents(query)

        # 构造历史对话,最多保留最近5轮
        messages = []
        for q, a in user_session['history'][-5:]:
            messages.append(HumanMessage(content=q))
            messages.append(AIMessage(content=a))

        # 重写问题以优化检索
        rewrite_prompt = f"""
        【文档】\n
        {doc_contents}\n

        【用户问题】\n
        {query}\n
        """

        callback_handler = WebSocketCallbackHandler(websocket)

        # 重写问题
        print("\n理解提问中...\n")
        await websocket.send("理解提问中...\n")

        rewrite_answer = ""
        async for chunk in self.llm.astream(
                [self.rewrite_sys_message] + messages + [HumanMessage(content=rewrite_prompt)]):
            print(chunk.content, end="", flush=True)
            rewrite_answer += chunk.content

        # 用提取的关键词再检索一次
        doc_contents = await self.retrieve_contents(rewrite_answer)

        # 构造完整Prompt
        full_prompt = f"""
                    【文档】\n
                    {doc_contents}\n

                    【用户问题】\n
                    {query}\n
                    """

        print("\n\n回答提问中...\n")
        await websocket.send("回答提问中...\n")
        answer = ""
        async for chunk in self.llm.astream([self.ask_sys_message] + messages + [HumanMessage(content=full_prompt)]):
            chunk_content = chunk.content
            if chunk_content:
                answer += chunk_content
                await callback_handler.on_llm_new_token(chunk_content)

        # 存入用户历史
        user_session['history'].append((query, answer))
        return answer
