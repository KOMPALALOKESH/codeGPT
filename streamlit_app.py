import streamlit as st
from llama_cpp import Llama

header = st.container()

header.title("🤖Chatbot")

header.markdown('''
**➡️CodeLlama: AI-powered chatbot companion.**  
**➡️Answering your questions about code, machine learning, and AI.**  
**➡️Helping you to learn and understand new concepts, Providing you with companionship and conversation.**  
''')

def load_llm():
  llm = Llama(model_path="codellama-13b-instruct.Q4_K_M.gguf", n_gpu_layers=50)
  return llm

def llm_response(prompt):
    llm = load_llm()
    llm_pipeline = llm(
       prompt,
       max_tokens= 2500
    )
    text_from_choices = llm_pipeline['choices'][0]['text']
    return text_from_choices

# def sample_code(prompt):
#     return f"user typed {prompt}"


def chat_actions():
    st.session_state["chat_history"].append(
        {"role": "user", 
        "content": st.session_state["chat_input"]},
    )

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": llm_response(st.session_state["chat_input"]),
        },
    )


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.chat_input("Enter your message", on_submit=chat_actions, key="chat_input")

for i in st.session_state["chat_history"]:
    with st.chat_message(name=i["role"]):
        st.write(i["content"])
