# AI Sentiment-Aligned Text Generator

##  Objective

This project implements an AI text generation system that dynamically adjusts its output based on the sentiment of the input prompt. It integrates sentiment analysis to classify the user's input (positive, negative, or neutral) and subsequently uses prompt engineering to condition a text generation model to produce an essay or paragraph aligned with the detected sentiment.

## 🛠️ Technical Approach and Methodology

### 1. Sentiment Analysis

** Model Used:** A pre-trained BERT-based model for sentiment classification (specifically, the \`distilbert-base-uncased-finetuned-sst-2-english\` variant, used via the \`sentiment-analysis\` pipeline in Hugging Face).  
**Classification:\*\* The model classifies the text as \`POSITIVE\` or \`NEGATIVE\`. A simple rule was implemented to classify text with a confidence score below 80% as \`NEUTRAL\`.  
**Framework:\*\* Hugging Face \`transformers\` library with PyTorch backend.

### 2. Text Generation

**Model Used:\*\* \*\*DistilGPT-2\*\* (a smaller, faster version of GPT-2). This model was chosen for its balance of generation quality and fast loading time, which is critical for a web demo.  
**Sentiment Alignment:\*\* Since DistilGPT-2 is not inherently sentiment-controllable, \*\*Prompt Engineering\*\* was used. The detected sentiment (e.g., "The overall mood should be \*\*positive\*\*.") is prepended to the user's prompt, guiding the model's tone for the generated output.  
**Framework:\*\* Hugging Face \`transformers\` library.

\#\#\# 3. Frontend Implementation

\* \*\*Framework:\*\* \*\*Streamlit\*\*. This framework was selected for its capability to rapidly build and deploy interactive Python-based web applications, fulfilling the requirement for a functional, user-friendly interface with minimal code overhead.

\#\#\# 4\. Dataset(s) Used

\* No custom dataset was used for training due to the 24-hour deadline. The models rely entirely on their \*\*pre-training data\*\* (e.g., BookCorpus, WebText, SST-2) as provided by the Hugging Face hub.

\#\# 🚀 Setup and How to Run the Project

1\.  \*\*Clone the Repository:\*\*  
    \`\`\`bash  
    git clone \[YOUR-REPO-LINK\]  
    cd ai-text-generator  
    \`\`\`  
2\.  \*\*Create a Virtual Environment (Recommended):\*\*  
    \`\`\`bash  
    python \-m venv venv  
    source venv/bin/activate  \# On Windows: venv\\Scripts\\activate  
    \`\`\`  
3\.  \*\*Install Dependencies:\*\*  
    \`\`\`bash  
    pip install \-r requirements.txt  
    \`\`\`  
4\.  \*\*Run the Streamlit Application:\*\*  
    \`\`\`bash  
    streamlit run app.py  
    \`\`\`  
5\.  \*\*Access:\*\* Open the displayed local URL (usually \`http://localhost:8501\`) in your web browser.

\#\# 🚧 Challenges Encountered and Reflections

\* \*\*Sentiment Control:\*\* The main challenge was ensuring the generated text strictly adheres to the detected sentiment, as general-purpose generative models like GPT-2 are not specialized for fine-grained sentiment control. \*\*Prompt engineering\*\* provided a pragmatic, yet imperfect, solution.  
\* \*\*Execution Time:\*\* Loading two large models (sentiment and generation) can be slow, affecting user experience. This was mitigated by using \`st.cache\_resource\` in Streamlit to ensure the models are loaded only once.  
\* \*\*Output Cleanliness:\*\* The output often included the input prompt or the sentiment prefix. Custom post-processing logic was added to parse and remove these artifacts to present a cleaner, more coherent paragraph/essay to the user.

\#\# ✨ Optional Enhancements Implemented

\* \*\*Manual Sentiment Selection:\*\* Users can override the automatic detection in the sidebar.  
\* \*\*Adjustable Length:\*\* A slider allows users to control the maximum length of the generated output.
