
# Provision ollama

```
install llama curl -fsSL https://ollama.com/install.sh | sh
start llama ollama serve
pull model ollama pull zephyr
```

create dirctory `/docs`

put your files into docs

# Create bot app

create file `app.py` with following code:

Source [link](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)

``` python
import os
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index import SimpleDirectoryReader
from llama_index.llms import Ollama

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
print(f"Connecting to ollama server {OLLAMA_HOST}")

# connect to ollama service running on OpenShift
# zephyr
my_llm = Ollama(model="zephyr", base_url="http://"+OLLAMA_HOST+":11434")

system_prompt = \
    "You are Service Bot, an expert on the Service and its functionality and your job is to answer questions about these two topics." \
    "Assume that all questions are related to Service and its functionality." \
    "Keep your answers to a few sentences and based on context – do not hallucinate facts." \
    "Always try to cite your source document."

st.title("ServiceBot")
st.subheader("Everything you want to know about Service")

if "messages" not in st.session_state.keys():
    # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the Service"}
    ]

@st.cache_resource(show_spinner=False)
def load_data(_llm):
    with st.spinner(text="Loading and indexing the document data – might take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=_llm, embed_model="local")
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data(my_llm)

chat_engine = index.as_chat_engine(
    chat_mode="context", verbose=True, system_prompt=system_prompt
)

if prompt := st.chat_input("Ask me a question about Service"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Querying..."):
            streaming_response = chat_engine.stream_chat(prompt)
            placeholder = st.empty()
            full_response = ''
            for token in streaming_response.response_gen:
                full_response += token
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            # Add response to message history
            st.session_state.messages.append(message)
            
```


# Dependencies

create file `requirements.txt`:
```
streamlit==1.27.2
torch==2.1.0
llama-index==0.9.2
transformers==4.35.2
pypdf
```

# Init virtualenv
```
python3 -m virtualenv venv --python=python3.11
actiavte virtualevnv source venv/bin/activate
python3 -m pip install -r requirements.txt
```

# Run
```
streamlit run code.py
```
