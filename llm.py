import ollama
from logger import LOGGER

logging = LOGGER().logging


class LLM:
    def __init__(self, model, chatmemorysize=20):
        self.chatmemorysize = chatmemorysize
        self.model = model

    def llm_clientstreamer(self, chat_memory, maxoutpuwords=None):
        """Sends prompt to Mistral model and returns response text"""
        stream = ollama.chat(
            model=self.model,
            messages=list(chat_memory),
            stream=True,
            options={"num_predict": maxoutpuwords},
        )

        for chunk in stream:
            yield chunk["message"]["content"]

    def generate(self, chat_memory):
        return "".join(chunk for chunk in self.llm_clientstreamer(chat_memory))
