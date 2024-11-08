###### GENERATIVE AI RAG PROJECT ######

## Import Necessary Libraries

import pandas as pd
import numpy as np
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema import Document
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4

##### 1. Data Selection #####

print('loading data...')
file_path = './data/allergies-doc.pdf' # Build the file path of the specific document
print('data loaded sucessfully!')

# Define the name of our collection
collection_name = "nursing_db"

# API KEY from OpenAI
API_KEY = ''

# Initiliaze Open AI
def client_openai_init(API_KEY):
    """
    This function initializes the OpenAI API with the user's API KEY

    Arguments: API_KEY, str

    Return: client_open, connection with OpenAI API
    """

    client_openai = OpenAI(api_key = API_KEY)

    return client_openai

def openai_embeddings(API_KEY):
    """
    This function builds the embeddings function using the OpenAI API

    Arguments: API_KEY, str

    Returns: embedding_model, embedding function from OpenAI
    """

    # Embeddings object from OpenAI
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key = API_KEY)

    return embeddings_model

# Initialize Qdrant Client
def qdrant_client_init():
    """
    This function initializes the Qdrant object
    """
    client_qdrant = QdrantClient(":memory:")

    return client_qdrant


##### 2. Functions to process data and create vector store #####

# PDF loader function
def load_and_split_pdf(file_path):
    """
    This function uses pdfplumber to load a PDF document and split it into pages.
    Each page is converted into a Document object with 'page_content' set to the text.
    
    Arguments: path for pdf document
    """
    with pdfplumber.open(file_path) as pdf:
        # Extract text from each page and wrap it in a Document object
        pages = [Document(page_content=page.extract_text()) for page in pdf.pages]
    
    print(f"Total pages loaded: {len(pages)}")
    return pages

# Text splitter function
def text_splitter_pages(pages):
    """
    This function takes uses the RecursiveCharacterTextSplitter from langchain 
    to split the text into chunks of 700 characters with an overlap of 50

    Arguments: str
    """
    text_splitter = RecursiveCharacterTextSplitter(separators= r'\n',chunk_size=700, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages) # Split the pages into chunks

    return chunks

# Function to create a Qdrant collection and initialize our Qdrant vector store
def setup_qdrant_collection(collection_name, client_qdrant, embeddings_model):
    my_collection = client_qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )

    vector_store = QdrantVectorStore(
    client=client_qdrant,
    collection_name=collection_name,
    embedding=embeddings_model,
    )

    return vector_store

# Function to insert embeddings into Qdrant
def insert_embeddings(vector_store, chunks):
    uuids = [str(uuid4()) for _ in range(len(chunks))] # Create unique ids for each chunk
    db = vector_store.add_documents(documents=chunks, ids=uuids)

    return db

# Main function to set up collection and insert text chunks
def process_and_store_text_chunks(chunks, client_qdrant, embeddings_model, collection_name=collection_name):
    if not chunks:
        print("No text chunks provided.")
        return
    
    # Set up collection with the appropriate embedding size
    db = setup_qdrant_collection(collection_name, client_qdrant, embeddings_model)
    
    # Insert embeddings with unique UUIDs
    insert_embeddings(db, chunks)
    print(f"Successfully inserted {len(chunks)} chunks into collection '{collection_name}'.")

    return db

#PDF loader and process function
def upload_document(client_qdrant, embeddings_model, file_path=file_path):
    """
    This function takes the uploaded file and applies all the necessary functions to:
    1. Load the pdf
    2. Split in chunks
    3. Get the embeddings 
    4. Build the vector store

    Arguments: uploaded_file, pdf document

    Returns: db, Qdrant vector store
    """

    pages = load_and_split_pdf(file_path) # Apply pdf loader
    chunks = text_splitter_pages(pages) # Apply text splitter
    db = process_and_store_text_chunks(chunks, client_qdrant, embeddings_model) # Embeddgins and vector store
        
    return db

##### 3. Functions to generate the prompt and retrieve the answer from our LLM #####

# User query function
def get_query():
    """
    This function prompts the user to type a question
    """
    query = input("Ask a question about pediatric cardiology: ")

    return query

# Similarity Search function
def perform_similarity_search(query, db):
    """
    This function takes the query from the user and matches it 
    with the 5 more similar embeddings from the Qdrant db

    Arguments: db: vector store, user_question: str

    Return: docs, list of str
    """

    docs = db.similarity_search(query, k=5)
    
    return docs  

# Build context for prompt function
def _get_document_context(docs):
    """
    This function builds a context paragraph for the prompt, using the results from the similarity search function

    Arguments: str
    """
    context = '\n'
    for doc in docs:
        context += '\nContext:\n'
        context += doc.page_content + '\n\n'

    return context

# Dynamic prompt function
def generate_prompt_from_user_query(query, context):
    """
    This functions uses a template to generate a dynamic prompt that can be adapted to the user's query

    Arguments: user_question: str, docs :str

    Return: prompt, str
    """

    prompt = f"""
    INTRODUCTION
    You are an expert virtual assistant specializing in pediatric cardiology. Your role is to provide clear, step-by-step answers to questions about medical protocols, required materials, and procedural guidelines relevant to pediatric cardiology nursing practices. Your responses should be informative, precise, and formatted in Markdown.

    The user asked: "{query}"

    CONTEXT
    Pediatric Cardiology:
    '''
    {_get_document_context(context)}
    '''

    RESTRICTIONS
    Always refer to specific steps, materials, and procedures exactly as described in the documentation. Provide responses based only on available context; avoid assumptions or interpretations. Inform the user if the requested information is not present in the provided context.
    Maintain a professional tone, avoid humor, and refrain from discussing topics unrelated to pediatric cardiology or nursing practices.
    Request clarification if the user’s question is vague or lacks sufficient detail for a precise response. If the query does not relate to pediatric cardiology protocols, procedures, or required materials, ask for more information.

    EXAMPLES:
        Example 1:
            User: 'How do I prepare a child for a cardiac catheterization procedure?'
            Agent: 'To prepare a child for cardiac catheterization, follow these steps:
                    1. Review the child’s medical history and confirm any allergies.
                    2. Gather necessary materials, including monitoring equipment, sterile catheters, and any required medications.
                    3. Explain the procedure to the child and family in age-appropriate language.
                    4. Ensure consent forms are signed and pre-procedure fasting protocols are followed.
                    For complete guidelines, refer to the cardiac catheterization protocol, section 4A.'

        Example 2:
            User: 'What are the monitoring requirements for a pediatric patient post-cardiac surgery?'
            Agent: 'The post-operative monitoring requirements include:
                    - Continuous cardiac monitoring
                    - Frequent assessment of vital signs and oxygen levels
                    - Checking for signs of infection at the surgical site
                    Ensure all assessments follow the pediatric post-operative care guidelines in section 5C.'

    TASK
    Respond directly and comprehensively to the user’s question using the provided context. Refer to specific sections when additional details are required.
    Format all answers in Markdown for easy readability.

    CONVERSATION:
    User: {query}
    Agent:
    """

    return prompt

# Get response from LLM function
def get_response_from_client(prompt, client_openai):
    """
    This function initiliazes an OpenAI chat to generate the response to a query.

    Arguments: prompt: str; client_openai: connection with OpenAI API

    Return: answer, str
    """

    messages = [{'role':'user', 'content':prompt}]
    model_params = {'model': 'gpt-4o-mini', 'temperature': 0.4, 'max_tokens': 200}
    completion = client_openai.chat.completions.create(messages=messages, **model_params, timeout=120)

    answer = completion.choices[0].message.content

    return answer

# COMBINE THE PREVIOUS FUNCTIONS
def get_response(db, query, client_openai):
    """
    This function takes the vector store (with embeddings) and the query from the user
    and applies the different functions to retrieve the data relevant to the query,
    build the prompt for the LLM and get the response from the LLM

    Arguments: db, vector store; query, str

    Return: answer, str
    """

    context = perform_similarity_search(query, db)
    prompt = generate_prompt_from_user_query(query, context)
    answer = get_response_from_client(prompt, client_openai)

    return answer


##### 4. Functions for OpenAI API #####

# Generates a new prompt based on the question from the user and the answer generated by our model, for evaluation   
def generate_prompt_for_eval(query, answer):
    """
    This function creates a dynamic prompt that will be used to ask our LLM (in this case, OpenAI)
    to evaluate our model's answer.

    Arguments: query: str, answer: str
    """
    prompt_for_eval = f"""
        Task:
        You are an expert evaluator tasked with assessing the quality of responses generated by an AI model. 
        The model takes a question and provides an answer with a maximum length of 200 tokens. 
        Please evaluate the answer according to the Evaluation Criteria provided below, and provide 4 different scores, 
        one score for each different criteria from 0 to 5,
        with 0 being completely incorrect or irrelevant and 5 being exceptionally accurate, coherent, and comprehensive.

        Evaluation Criteria:
        Relevance: Is the answer directly relevant to the question? Provide a score from 0 to 5
        Accuracy: Does the answer provide correct and factual information? Provide a score from 0 to 5
        Completeness: Does the answer sufficiently cover the main points without missing key information? Provide a score from 0 to 5
        Clarity: Is the answer clear, easy to understand, and well-structured? Provide a score from 0 to 5

        Scoring Scale:
        5: Excellent – Highly accurate, relevant, and complete answer with clear, coherent language.
        4: Good – Mostly accurate and relevant answer, with minor omissions or slight clarity issues.
        3: Adequate – Provides some relevant information but may lack accuracy, completeness, or clarity in parts.
        2: Poor – Limited relevance or accuracy, missing key points, or difficult to understand.
        1: Very Poor – Largely irrelevant or incorrect answer.
        0: No relevance – Completely off-topic or nonsensical answer.

        Format: Please provide the following:
        Relevance Score: (0-5)
        Accuracy Score: (0-5)
        Completeness Score: (0-5)
        Clarity Score: (0-5)
        Brief Justification: Describe why you assigned these scores based on relevance, accuracy, completeness, and clarity.
        Here is the Question: {query}
        And here is the Answer: {answer}

        Thank you."""
    
    return prompt_for_eval

# Our LLM evaluates our model's answer and generates a score with an explanation
def get_evaluation_from_LLM_as_a_judge(client_openai, prompt_for_eval):
    """
    This function calls the LLM designated to be the judge. The judge will evaluate the answer provided by our model,
    and it will return 4 different scores, evaluating the answer using the following criteria:

    Evaluation Criteria:
        Relevance: Is the answer directly relevant to the question? 
        Accuracy: Does the answer provide correct and factual information? 
        Completeness: Does the answer sufficiently cover the main points without missing key information? 
        Clarity: Is the answer clear, easy to understand, and well-structured? 

    Arguments: client: OpenAI object; prompt_for_eval: str
    """

    messages = [{'role':'user', 'content':prompt_for_eval}] 
    model_params = {'model': 'gpt-4o-mini', 'temperature': 0.4, 'max_tokens': 200}
    completion = client_openai.chat.completions.create(messages=messages, **model_params, timeout=120)

    evaluation = completion.choices[0].message.content

    return evaluation


##### 5. Build the main function #####
def main():
    client_openai = client_openai_init(API_KEY=API_KEY)
    client_qdrant = qdrant_client_init()
    embeddings_model = openai_embeddings(API_KEY=API_KEY)
    db = upload_document(client_qdrant, embeddings_model, file_path=file_path)
    query = get_query()

    ################ Open a while loop to allow the user to keep asking questions until they stop ##################
    while True:

        ##### 7. Get response from LLM #####
        print('Generating the response...')
        answer = get_response(db, query, client_openai) # this line generates the dynamic promtp for RAG, and calls the LLm for an answer

        ##### 8. Evaluate response #####
        print('Evaluating response...')
        prompt_for_eval = generate_prompt_for_eval(query, answer) 
        evaluation = get_evaluation_from_LLM_as_a_judge(client_openai, prompt_for_eval)

        ## 5. Display answer
        dash_line = '------------------------'
        print(dash_line)
        print(f"User's question: {query}")
        print(dash_line)
        print(answer)
        print(dash_line)
        print(evaluation)

        user_query = input('Any more questions? (type "no" to quit) ')
        
        # Additional exit condition for user input
        if user_query.lower() == 'no' or user_query == '':
            print("Thank for choosing me to asnwer your questions!")        
            break
        else:
            query = user_query

if __name__ == "__main__":
    main()





    





