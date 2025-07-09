from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler


class WebSocketCallbackHandler(AsyncCallbackHandler):

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        try:
            print(token, end="", flush=True)
            await self.websocket.send(token)
        except Exception as e:
            print(f"发送token出错: {e}")
