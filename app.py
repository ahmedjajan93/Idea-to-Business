import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables (for OpenAI API key)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Streamlit UI ---
st.set_page_config(page_title="Startup Blueprint Generator", layout="centered")
st.title("🚀 Idea-to-Business ")
st.markdown("Turn your startup idea into a full plan using AI.")


# Initialize ChatOpenAI LLM
@st.cache_resource
def load_model():
    llm = ChatOpenAI(
        model="deepseek/deepseek-chat:free",
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,  # Balance creativity vs. consistency (0=strict, 1=creative)
    )
    return llm


llm = load_model()


# Prompt template for generating startup blueprint
blueprint_prompt = PromptTemplate(
    input_variables=["idea", "language", "arab_countries"],
    template="""
You are a startup expert and strategist. A user has the following idea:

"{idea}"
translate the reslut to {language}. 

Follow these rules:
1. **Language**: {language} (must be natural and idiomatic)
2. **Country**: {arab_countries} (based on the user's location and courncey)


Based on this, generate a complete startup blueprint with these sections:
1. Problem the idea solves
2. Target audience / customer personas
3. Value proposition
4. Core features / MVP plan
5. Monetization strategies
6. Competitive edge
7. Cost structure / budget
8. One-sentence pitch

Respond in a clear, structured format.
""",
)

# Create LLMChain
chain = LLMChain(llm=llm, prompt=blueprint_prompt)


idea_input = st.text_area(
    "💡 Describe your startup idea",
    height=200,
    placeholder="Example: An AI-powered app that helps people meal plan based on what's in their fridge.",
)
language = st.selectbox(
    "🌐 Language",
    [
        "English",
        "Arabic",
    ],
    help="Target language for translation/rewriting.",
)
arab_countries = st.selectbox(
    "📍 Country",
    [
        "Algeria",
        "Bahrain",
        "Comoros",
        "Djibouti",
        "Egypt",
        "Iraq",
        "Jordan",
        "Kuwait",
        "Lebanon",
        "Libya",
        "Mauritania",
        "Morocco",
        "Oman",
        "Palestine",
        "Qatar",
        "Saudi Arabia",
        "Somalia",
        "Sudan",
        "Syria",
        "Tunisia",
        "United Arab Emirates",
        "Yemen",
    ],
    help="Target language for translation/rewriting.",
)

if st.button("📊 Generate Blueprint"):
    if idea_input:
        with st.spinner("Thinking like a founder..."):
            response = chain.run(
                {
                    "idea": idea_input,
                    "language": language,
                    "arab_countries": arab_countries,
                }
            )
            st.write(response)
        st.markdown("### 📘 Business Blueprint")
        st.success(response.strip())
    else:
        st.warning("Please describe your idea first.")
