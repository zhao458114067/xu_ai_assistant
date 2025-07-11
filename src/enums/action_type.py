from enum import Enum


class ActionTypeEnums(Enum):
    CONNECT = (10000, "连接")
    ASK = (10001, "提问")
    INTERRUPT = (10002, "打断")

    def __init__(self, code, message):
        self._code = code
        self._message = message

    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self._message
