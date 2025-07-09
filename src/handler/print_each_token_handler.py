from langchain.callbacks.base import BaseCallbackHandler

class PrintEachTokenHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
