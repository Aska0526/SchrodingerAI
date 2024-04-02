__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from datasets import load_dataset
import streamlit as st
from Gemma2BInstructModel import Gemma2BInstructModel
from VectorStore import VectorStore
import pandas as pd

# Title of the web app
st.set_page_config(
   page_title='You are talking to Schrodinger',
   page_icon='⚛️',  # You can use an emoji or a path to an image file
)

st.title('Schrodinger-AI -- Using Nothing!'   #Using Google\'s Gemma 2B')
st.text('I am Schrodinger back to life! hehe...')
st.image('./head.png')
link = 'https://www.linkedin.com/in/chao-fan-818667245/'
st.link_button("follow me here", link)


# Load only the training split of the dataset
train_dataset = load_dataset("camel-ai/physics", split='train')

# Filter the dataset to only include entries with the 'closed_qa' category
closed_qa_dataset = train_dataset.filter(lambda example: example['topic;'] == 'Quantum mechanics')

vector_store = VectorStore("quantum-mechanics-knowledge-base")
    
 # Assuming closed_qa_dataset is defined and available
vector_store.populate_vectors(closed_qa_dataset)

#gemma_model = Gemma2BInstructModel()

###################################################################################

def get_response(prompt:str):
    # Fetch context from VectorStore, assuming it's been populated
    context_response = vector_store.search_context(prompt)

    # Extract the context text from the response
    # The context is assumed to be in the first element of the 'context' key
    context = "".join(context_response['documents'][0])
    return(context)
    #return gemma_model.generate_answer(prompt,context=context)

#####################################################################################
# User input
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me something."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_response(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


###knowledge base
if st.checkbox('Show my knowledge base here'):
    df_data = pd.DataFrame(closed_qa_dataset)[['sub_topic','message_1','message_2']]
    st.write(df_data)