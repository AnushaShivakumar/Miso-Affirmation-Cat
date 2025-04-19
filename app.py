import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from streamlit_lottie import st_lottie

import openai

import torch
import base64
import json

# --- App Config ---
st.set_page_config(page_title="Miso the Affirmation Cat", page_icon="🐾", layout="centered")

# --- Load API Key from secrets.toml ---
openai.api_key = st.secrets["openai"]["api_key"]


# --- Load Custom Font ---
def load_font_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

winky_font_b64 = load_font_base64("WinkyRough-VariableFont_wght.ttf")

# --- Lottie Loader ---
def load_lottie(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        st.warning("Couldn't load cat animation.")
        return None

cat_lottie = load_lottie("cat_animation.json")

# --- Custom CSS ---
st.markdown(f"""
    <style>
    @font-face {{
        font-family: 'WinkyRough';
        src: url(data:font/ttf;base64,{winky_font_b64}) format('truetype');
    }}

    html, body, h1, h2, h3, .stMarkdown, .stTextInput, .stSubheader {{
        font-family: 'WinkyRough', cursive !important;
        background: linear-gradient(145deg, #fff1f8, #e6f6ff);
        color: #4b244a;
    }}

    .block-container {{
        max-width: 700px;
        margin: auto;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}


    .stTextInput > div > div > input {{
        background-color: #ffffff;
        color: #4b244a;
        border-radius: 12px;
        padding: 14px;
        font-size: 1.1rem;
        border: 1px solid #b197bd;
        box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.05);
    }}

    .stButton > button {{
        background-color: #ffc8dd;
        color: #4b244a !important;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 28px;
        border-radius: 30px;
        border: none;
        transition: all 0.3s ease-in-out;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
    }}

    .stButton > button:hover {{
        background-color: #b197bd;
        color: #ffffff !important;
        transform: scale(1.03);
    }}

    .stAlert {{
        border-radius: 12px !important;
        padding: 1rem !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Load GPT-2 & Emotion Model ---
@st.cache_resource
def load_gpt2():
    return GPT2Tokenizer.from_pretrained("gpt2"), GPT2LMHeadModel.from_pretrained("gpt2")

@st.cache_resource
def load_emotion():
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    return tokenizer, model

tokenizer, gpt2_model = load_gpt2()
emo_tokenizer, emo_model = load_emotion()

# --- Emotion Detection ---
def detect_emotion(text):
    inputs = emo_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = emo_model(**inputs).logits
    return emo_model.config.id2label[torch.argmax(logits).item()]

# --- Local Fallback Generator ---
from transformers import pipeline

# Initialize generator only once
generator = pipeline("text-generation", model="gpt2")

def generate_local_affirmation(emotion):
    fallback_prompt = (
        f"You are Miso, a sassy cat giving a pep talk. A human is feeling {emotion}. "
        f"Write one clever, supportive sentence. Be warm, short, and witty."
    )
    inputs = tokenizer.encode(fallback_prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=1.0,
        top_k=40,
        top_p=0.95,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt and keep just the first clean sentence
    response = result.replace(fallback_prompt, "").strip()
    first_line = response.split(".")[0].strip()
    
    # Return with punctuation
    return first_line + "." if first_line else "You're doing great. Really."

# --- OpenAI Affirmation Generator ---


def generate_affirmation(name, emotion, tone_prefix):
    prompt = (
        f"{tone_prefix} Your friend {name} is feeling {emotion}."
        " Respond with a comforting one-liner as Miso the Affirmation Cat: sassy, smart, and warm (but not cringey)."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Miso, a cat that gives smart, witty, kind affirmations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60,
            temperature=0.9,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        st.warning("⚠️ OpenAI quota exceeded or an error occurred. Please try again later!")
        return "Miso says: I'm taking a little nap right now. Try again in a moment!"

# --- UI ---
st.title("🐾 Miso the Affirmation Cat")
st.subheader("Drop your vibe. Miso turns it into purr therapy with a touch of snark. 💭")
if cat_lottie:
    st_lottie(cat_lottie, height=200)

name = st.text_input("What's your name?", placeholder="e.g., Anusha")
mood = st.text_input("How are you feeling today?", placeholder="e.g., tired but hopeful")

if st.button("Talk to Miso"):
    if mood:
        with st.spinner("Miso is thinking..."):
            emotion = detect_emotion(mood)

            if emotion.lower() in ["sadness", "fear", "anxiety"]:
                tone_prefix = "Offer gentle comfort like a wise, purring friend."
            elif emotion.lower() in ["anger", "frustration"]:
                tone_prefix = "Offer something grounding with a touch of snark."
            elif emotion.lower() in ["joy", "gratitude", "calm"]:
                tone_prefix = "Celebrate the moment with playful kindness."
            else:
                tone_prefix = "Offer a neutral yet warm affirmation."

            affirmation = generate_affirmation(name or "friend", emotion, tone_prefix)
            if affirmation:
                st.success(f"🐾 Miso: Hi {name.strip().title()}, \"{affirmation}\"")

            else:
                st.warning("⚠️ OpenAI quota exceeded or error occurred. Falling back to local Miso 😽")
                fallback = generate_local_affirmation(emotion)
                st.info(f"🐱 Miso (offline mode): \"{fallback}\"")
    else:
        st.warning("Please share how you're feeling first!")
