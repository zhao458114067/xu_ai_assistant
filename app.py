import asyncio

import websockets
from aiohttp import web

from src.ai_assistant import AIAssistantService

ai_assistant = AIAssistantService()


async def health_http_handler(request):
    return web.Response(text="healthy")


async def websocket_handler(websocket):
    sid = f"ws_{id(websocket)}"
    print(f"连接已建立，SID: {sid}")

    try:
        for i in range(100):
            message = await websocket.recv()
            print(f"收到来自 {sid} 的消息: {message}")

            user_session = ai_assistant.get_user_session(sid)
            # # 中断请求
            if message.strip().lower() in ["stop", "cancel"]:
                if user_session["task"] and not user_session["task"].done():
                    user_session["task"].cancel()
                    user_session["active"] = False
                    await websocket.send("当前任务已被取消")
                else:
                    await websocket.send("当前无可取消的任务")
                continue

            # 已在处理中
            if user_session["active"]:
                await websocket.send("请等待当前回答完成或发送 'stop' 中断")
                continue

            # 开始处理新问题
            task = asyncio.create_task(ai_assistant.process_query(sid, message, websocket))
            user_session["task"] = task
            user_session["active"] = True

            def unlock_session(task_result):
                user_session["active"] = False

            task.add_done_callback(unlock_session)

    except Exception as e:
        print(f"WebSocket错误 ({sid}): {e}")
    finally:
        user_session = ai_assistant.get_user_session(sid)
        task = user_session["task"]
        if task and not task.done():
            task.cancel()
        print(f"连接断开，SID: {sid}")


async def main():
    # 创建HTTP服务器
    http_app = web.Application()
    http_app.router.add_get('/health', health_http_handler)
    runner = web.AppRunner(http_app)
    await runner.setup()
    http_site = web.TCPSite(runner, '0.0.0.0', 19000)
    await http_site.start()

    # 创建WebSocket服务器
    async with websockets.serve(websocket_handler,
                                "0.0.0.0",
                                19001,
                                ping_interval=60,
                                ping_timeout=30):
        print("""
        AI助手服务已启动:
        HTTP server running at http://0.0.0.0:19000
        WebSocket server running at ws://0.0.0.0:19001
        """)
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())
