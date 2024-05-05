from Embeddings import Embeddings
import yaml
from llm import LLM
from logger import LOGGER

logging = LOGGER().logging

userdict = lambda prompt: {"role": "user", "content": prompt}
sysdict = lambda prompt: {"role": "system", "content": prompt}
assistdict = lambda prompt: {"role": "assistant", "content": prompt}


class Chatbot:
    def __init__(self, config, vectordb):
        self.vectorDB = vectordb
        self.config = config
        self.llm = LLM(config["model"])

    def reflect(self, question, conversation_history):

        prompt = self.config["rag_necessity_prmopt"].format(query=question)
        response = self.llm.generate(conversation_history + [sysdict(prompt)])
        logging.debug("Response after reflecting >>>>>>>>>>>>>>>>>>>>>\n %s", response)
        if '"no"' in response.lower():
            logging.info("Skipping RAG")
            return None
        else:
            logging.info("Will use RAG")
            return response

    def create_sys_prompt(self, question, rag_context=None):

        if rag_context is None:
            prompt = self.config["qa_system_prompt_without_rag"].format(query=question)
        else:
            prompt = self.config["qa_system_prompt_with_rag"].format(
                rag_context=rag_context, query=question
            )
        return prompt

    def query_vectorDB(self, query):
        return self.vectorDB.search(query)

    def respond_with_streamer(self, question, conversation_history, maxoutpuwords):

        rag_query_optional = self.reflect(question, conversation_history)

        if rag_query_optional is None:
            sys_prompt = self.create_sys_prompt(question)
        else:
            logging.info("Performing RAG search ...")
            context = self.query_vectorDB(rag_query_optional)
            sys_prompt = self.create_sys_prompt(question, context)

        logging.debug(">>> sysprompt:\n%s", sys_prompt)

        return self.llm.llm_clientstreamer(
            conversation_history + [sysdict(sys_prompt)], maxoutpuwords
        )

    def stream(self, question, conversation_history, session_id, maxoutpuwords):

        logging.debug("%s <<<< %s", session_id, question)

        response = ""
        for chunk in self.respond_with_streamer(
            question, conversation_history, maxoutpuwords
        ):
            yield chunk
            response += chunk

        logging.debug("%s >>>> %s", session_id, response)


# Example usage
if __name__ == "__main__":

    CONFIG = yaml.load(open("config.yaml"), yaml.SafeLoader)
    retriever = Embeddings(CONFIG["embeddings"]).as_retriever()
    chatbot = Chatbot(CONFIG["llm"], retriever)
    question = "Is it going to rain today?"
    answer = chatbot.generate(question)
    print(answer)
