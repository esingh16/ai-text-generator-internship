import streamlit as st
from transformers import pipeline, set_seed
import requests # <-- NEW: Import for API calls

# Set a fixed seed for reproducibility in text generation (optional but good practice)
set_seed(42)

# --- Load Models (Cached for efficiency) ---
@st.cache_resource
def load_generator_model():
    # Load ONLY the generation model locally to save memory
    return pipeline("text-generation", model="distilgpt2")

# IMPORTANT: Call the loading functions after the definitions
text_generator = load_generator_model()


# --- Main Application Logic ---

def get_sentiment(text):
    """Classifies the sentiment of the input text using Hugging Face's Inference API."""
    
    # Define the API endpoint and access the secret key
    API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
    # Accesses the secret securely from the Streamlit Cloud dashboard (see Step 2)
    API_KEY = st.secrets.get("HUGGINGFACE_API_KEY") 

    if not API_KEY:
        st.error("HUGGINGFACE_API_KEY not found in secrets. Using 'neutral' fallback.")
        return "neutral"
        
    headers = {"Authorization": f"Bearer {API_KEY}"}

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        response.raise_for_status() # Raise an exception for bad status codes
        
        # The API returns a list of lists of dictionaries
        result = response.json()[0]
        label = result[0]['label'].lower()
        score = result[0]['score']

        # Custom Neutral rule for better classification
        if score < 0.75:
             return 'neutral'
        return label
    except requests.exceptions.RequestException as e:
        st.warning(f"API request failed: {e}. Using 'neutral' fallback.")
        return "neutral"
    except Exception:
        return "neutral"


def generate_text(prompt, sentiment, max_length=150, num_return_sequences=1):
    """Generates text based on the prompt and the detected sentiment."""

    sentiment_prefix = f"The overall mood should be **{sentiment}**. "
    full_prompt = sentiment_prefix + prompt

    generated_list = text_generator(
        full_prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.9,
        pad_token_id=text_generator.tokenizer.eos_token_id
    )
    
    full_output = generated_list[0]['generated_text']

    if sentiment_prefix in full_output:
        final_text = full_output.split(sentiment_prefix, 1)[-1].strip()
        if prompt in final_text:
             return final_text.split(prompt, 1)[-1].strip()
        return final_text

    return full_output


# --- Streamlit Frontend UI ---

st.title("🤖 AI Sentiment-Aligned Text Generator")
st.markdown("Enter a topic or a seed sentence. The AI will first detect the sentiment (via API) and then generate a paragraph or essay matching that feeling.")
st.markdown("---")


# 1. User Input Area
user_prompt = st.text_area(
    "📝 Enter your prompt (e.g., 'The future of space travel is exciting' or 'Write about climate change'):",
    "The new advancements in AI are truly astonishing.",
    height=150
)

# 2. Optional Enhancements (User Selection)
st.sidebar.header("⚙️ Generation Settings")
# Manual sentiment selection
manual_sentiment = st.sidebar.radio(
    "Override Detected Sentiment (Optional):",
    ('Auto-Detect', 'positive', 'negative', 'neutral')
)

# Length adjustment
max_len = st.sidebar.slider(
    "Max Output Length (Tokens):",
    min_value=50,
    max_value=300,
    value=150,
    step=25
)


# 3. Execution Button
if st.button("✨ Generate Text"):
    if not user_prompt.strip():
        st.error("Please enter a prompt to generate text.")
    else:
        # Status message
        with st.spinner('Analyzing sentiment and generating text...'):

            # --- Sentiment Detection ---
            if manual_sentiment == 'Auto-Detect':
                detected_sentiment = get_sentiment(user_prompt)
            else:
                detected_sentiment = manual_sentiment

            st.subheader(f"✅ Detected/Selected Sentiment: **{detected_sentiment.upper()}**")
            st.markdown("---")

            # --- Text Generation ---
            generated_text = generate_text(
                user_prompt,
                detected_sentiment,
                max_length=max_len
            )

            # --- Output Display ---
            st.subheader("Generated Text")
            st.success(generated_text)

            # Display metadata for clarity
            st.info(f"**Seed Prompt:** {user_prompt}\n\n**Generation Model:** DistilGPT-2")
