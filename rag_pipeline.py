import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables (Google API Key)
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def fetch_transcript(video_id):
    """
    Fetches transcript for the given YouTube video ID.
    If the transcript is unavailable (e.g., no captions), returns None.
    """
    try:
        # Fetch the transcript in English or German if available
        en_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en', 'de'])
        # Join all caption snippets into one text string
        transcript_text = " ".join(snippet.text for snippet in en_transcript)
        return transcript_text
    except TranscriptsDisabled:
        return None


def create_vector_store(transcript_text):
    """
    Splits the transcript into chunks and creates a FAISS vector store
    for semantic search using Google Generative AI embeddings.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([transcript_text])

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


def get_answer(vector_store, query):
    """
    Retrieves the most relevant transcript chunks using FAISS retriever
    and answers the user's query using Gemini LLM.
    """
    # Retrieve top 4 most similar transcript chunks
    retrieval = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retrieved_docs = retrieval.invoke(query)

    # Initialize Google Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

    # Prompt template ensures answers are based on the transcript only
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        ANSWER ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        Context: {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    # Merge retrieved docs into one context string
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.format(context=context_text, question=query)
    answer = llm.invoke(final_prompt)

    return answer.content


def get_summary(vector_store):
    """
    Generates a full summary of the video using Gemini LLM.
    It uses a larger chunk of the transcript to create a concise summary.
    """
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retrieval = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

    prompt = PromptTemplate(
        template="""
        Summarize the following transcript in a concise and clear way:
        {context}
        """,
        input_variables=["context"]
    )

    parallel_chain = RunnableParallel({
        'context': retrieval | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    return main_chain.invoke("Summarize the video")
