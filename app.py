from logger import LOGGER

logging = LOGGER().logging

from streamlit_chat import message
import streamlit as st
from chatbot import Chatbot, userdict, assistdict
from langchain.globals import set_debug
import yaml
from Embeddings import Embeddings
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

st.set_page_config(layout="wide")
st.title("Experimental Medical Chatbot")


@st.cache_resource
def init():
    set_debug(True)
    CONFIG = yaml.load(open("config.yaml"), yaml.SafeLoader)
    chroma_db = Embeddings(CONFIG["embeddings"])
    return CONFIG, Chatbot(CONFIG["llm"], chroma_db)


def displaymsg(config):
    if "messages" not in st.session_state:
        st.session_state["messages"] = [assistdict(config["greeting"])]
    col1, col2, col3 = st.columns([1, 1, 1])
    if col1.button("Clear messages"):
        while st.session_state["messages"]:
            st.session_state["messages"].pop()
    if col2.button("Delete last message"):
        if st.session_state["messages"]:
            st.session_state["messages"].pop()
    if col3.button("Refresh"):
        pass
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main():
    config, chatbot = init()
    session_id = get_script_run_ctx().session_id
    maxoutpuwords = st.slider(
        "Maximum output words", min_value=50, max_value=1000, value=200
    )
    displaymsg(config)
    if prompt := st.chat_input("Chat here"):
        logging.debug(str(st.session_state["messages"]))
        logging.info(
            "\n\n******************************* User Session: %s ***********************************",
            session_id,
        )
        st.chat_message("user").markdown(prompt)
        st.session_state["messages"].append(userdict(prompt))
        with st.spinner("Generating response..."):
            with st.chat_message("assistant"):
                response = ""
                placeholder = st.empty()
                for chunk in chatbot.stream(
                    prompt, st.session_state["messages"][:-1], session_id, maxoutpuwords
                ):
                    response += chunk
                    placeholder.write(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        logging.info(
            "******************************* User Session: %s ***********************************\n\n",
            session_id,
        )


if __name__ == "__main__":
    main()
