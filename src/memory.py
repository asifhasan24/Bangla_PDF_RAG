from collections import deque

class ChatMemory:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.history = deque()

    def add_user_message(self, text: str):
        self.history.append(f"User: {text}")
        self._trim()

    def add_bot_message(self, text: str):
        self.history.append(f"Assistant: {text}")
        self._trim()

    def _trim(self):
        while len(self.history) > self.max_messages:
            self.history.popleft()

    def get_context(self) -> str:
        return "\n".join(self.history)
