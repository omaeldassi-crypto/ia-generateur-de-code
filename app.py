import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import os
import datetime

# -----------------------------------------------------
# ⚙️ CONFIGURATION DE LA PAGE
# -----------------------------------------------------
st.set_page_config(
    page_title="🤖 Assistant Multi-Capacités Léger",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------------------------------
# 🧠 EN-TÊTE DE L'APPLICATION
# -----------------------------------------------------
st.title("🧠 Assistant IA Léger – GPT-2 Edition")
st.caption("💡 Chatbot capable de **générer du texte, du code et simuler des images** — compatible CPU.")
st.divider()

# -----------------------------------------------------
# 🔁 CHARGEMENT DU MODÈLE GPT-2
# -----------------------------------------------------
@st.cache_resource
def load_generator():
    """Charge le modèle GPT-2 et le garde en cache."""
    return pipeline("text-generation", model="gpt2")

generator = load_generator()

# -----------------------------------------------------
# 🖼️ GÉNÉRATION D’IMAGE SIMULÉE
# -----------------------------------------------------
def generer_image(prompt_image, output_filename):
    """Simule la génération d'image (placeholder)."""
    try:
        img = Image.new('RGB', (512, 512), color=(40, 40, 60))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except IOError:
            font = ImageFont.load_default()
        d.text((20, 230), f"Simulation d'image :\n{prompt_image}", fill=(255, 255, 100), font=font)
        img.save(output_filename)
        return True
    except Exception as e:
        st.error(f"Erreur d'image : {e}")
        return False

# -----------------------------------------------------
# 💬 INITIALISATION DE LA CONVERSATION
# -----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.memory = []  # mémoire courte
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Bonjour ! Je suis un assistant IA léger. "
                   "Tapez une question, une commande `!image`, ou du code à générer."
    })

# -----------------------------------------------------
# 🔄 AFFICHAGE DES MESSAGES
# -----------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------------------------
# 💡 ENTRÉE UTILISATEUR
# -----------------------------------------------------
if prompt := st.chat_input("💬 Écrivez votre message ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # -------------------------------------------------
    # 🧩 GÉNÉRATION DE RÉPONSE
    # -------------------------------------------------
    with st.chat_message("assistant"):
        with st.spinner("L’IA réfléchit..."):
            réponse_finale = ""

            # 🖼️ Commande spéciale : génération d'image simulée
            if prompt.lower().startswith("!image"):
                prompt_image = prompt[6:].strip() or "Aucune description fournie"
                filename = f"image_{datetime.datetime.now().strftime('%H%M%S')}.png"
                st.info(f"🎨 Simulation d'image pour : **{prompt_image}**")

                if generer_image(prompt_image, filename):
                    st.image(filename, caption=f"Image simulée : {prompt_image}")
                    réponse_finale = "Voici votre image simulée. (💡 La vraie génération nécessite un GPU)"
                    os.remove(filename)
                else:
                    réponse_finale = "⚠️ Impossible de générer l'image simulée."

            # 🧠 Commande spéciale : mémoire
            elif prompt.lower().startswith("!mémoire"):
                mémoire_text = "\n".join([f"- {m}" for m in st.session_state.memory[-5:]]) or "Mémoire vide."
                réponse_finale = f"🧠 **Mémoire récente :**\n{mémoire_text}"

            # 💬 Réponse GPT-2 (texte / code / discussion)
            else:
                try:
                    input_text = " ".join(st.session_state.memory[-3:]) + " " + prompt
                    response = generator(
                        input_text,
                        max_length=200,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.95
                    )[0]['generated_text']

                    # Nettoyage
                    if response.startswith(prompt):
                        response = response[len(prompt):].strip()

                    réponse_finale = response
                    st.session_state.memory.append(prompt)  # stocke la mémoire courte
                except Exception as e:
                    st.error(f"Erreur du modèle : {e}")
                    réponse_finale = "❌ Erreur de génération de texte."

        st.markdown(réponse_finale)
        st.session_state.messages.append({"role": "assistant", "content": réponse_finale})

# -----------------------------------------------------
# 🎛️ PIED DE PAGE
# -----------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.9em;'>
Propulsé par 🤗 Hugging Face | Conçu pour CPU | Interface Streamlit améliorée ✨
</div>
""", unsafe_allow_html=True)
