import streamlit as st
from rag_pipeline import fetch_transcript, create_vector_store, get_answer, get_summary, extract_video_id

# Streamlit App Title
st.title("🎥 YouTube Video Q&A and Summary Bot")
st.write("Enter a YouTube video link to fetch its transcript, then ask a question or get a summary.")

# Input field for YouTube Video Link
video_url = st.text_input("Enter YouTube Video URL:", "https://www.youtube.com/watch?v=Gfr50f6ZBvo")

# Fetch transcript when button is clicked
if st.button("Fetch Transcript"):
    try:
        transcript_text = fetch_transcript(video_url)
        if transcript_text is None:
            st.error("No captions available for this video.")
        else:
            # Create FAISS vector store from transcript and store it in session state
            st.session_state.vector_store = create_vector_store(transcript_text)
            st.success("Transcript fetched and processed successfully!")
    except ValueError as e:
        st.error(str(e))

# After fetching transcript, show options
if "vector_store" in st.session_state:
    st.subheader("Choose an Option")
    option = st.radio("What would you like to do?", ["Get Full Summary", "Ask a Question"])

    # Option 1: Full Summary
    if option == "Get Full Summary":
        if st.button("Generate Summary"):
            summary = get_summary(st.session_state.vector_store)
            st.subheader("Video Summary:")
            st.write(summary)

    # Option 2: Ask Question
    elif option == "Ask a Question":
        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            answer = get_answer(st.session_state.vector_store, query)
            st.subheader("Answer:")
            st.write(answer)
