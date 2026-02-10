import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
import os
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

nltk.download("punkt", download_dir="nltk_data")
nltk.download("stopwords", download_dir="nltk_data")


# Load a pre-trained Hugging Face model (once, to optimize performance)
@st.cache_resource
def load_chatbot_model():
    try:
        return pipeline("text-generation", model="distilgpt2")
    except Exception as e:
        st.error(f"Error loading the chatbot model: {e}")
        return None

chatbot = load_chatbot_model()

# Define healthcare-specific response logic
def healthcare_chatbot(user_input):
    user_input_lower = user_input.lower()

    # Rule-based responses
    keywords_responses = {
        "symptom": "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice.",
        "appointment": "Would you like me to schedule an appointment with a doctor?",
        "medication": "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor.",
        "emergency": "If this is an emergency, please call emergency services immediately!",
        "prescription": "For prescriptions, it's best to follow your doctor’s advice. Let me know if you need general information."
    }

    for keyword, response in keywords_responses.items():
        if keyword in user_input_lower:
            return response

    # Generate response using Hugging Face model
    if chatbot:
        try:
            response = chatbot(user_input, max_length=50, num_return_sequences=1)
            return response[0]['generated_text']
        except Exception as e:
            return f"An error occurred while generating a response: {e}"

    return "I'm sorry, I couldn't process your request at the moment."

# Streamlit web app interface
def main():
    st.title("🩺 Healthcare Assistant Chatbot")
    st.write("Welcome! Ask me anything about healthcare, symptoms, medication, and appointments.")

    # Initialize session state for conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # User input
    user_input = st.text_input("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input.strip():
            # Append user query to conversation
            st.session_state.conversation.append(("User", user_input))

            # Get chatbot response
            response = healthcare_chatbot(user_input)

            # Append chatbot response to conversation
            st.session_state.conversation.append(("Healthcare Assistant", response))
        else:
            st.warning("Please enter a query.")

    # Display conversation history with improved formatting
    st.subheader("Conversation History")
    for role, message in st.session_state.conversation:
        with st.chat_message(role):
            st.write(message)

if __name__ == "__main__":
    main()
