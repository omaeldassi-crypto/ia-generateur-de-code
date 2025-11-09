import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import io
import datetime

# -----------------------------------------------------
# ⚙️ CONFIGURATION DE LA PAGE
# -----------------------------------------------------
st.set_page_config(
    page_title="🤖 Assistant IA Léger (GPT-2)",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------------------------------
# 🧠 TITRE ET DESCRIPTION
# -----------------------------------------------------
st.title("🧠 Assistant IA Léger – GPT-2 Edition (CPU Friendly)")
st.caption("💡 Chatbot textuel & simulateur d’images – 100 % compatible Streamlit Cloud (CPU only).")
st.divider()

# -----------------------------------------------------
# 🔁 CHARGEMENT DU MODÈLE GPT-2
# -----------------------------------------------------
@st.cache_resource
def load_generator():
    """Charge le modèle GPT-2 et le garde en cache pour éviter de le recharger."""
    return pipeline("text-generation", model="gpt2")

generator = load_generator()

# -----------------------------------------------------
# 🖼️ SIMULATION D’IMAGE (EN MÉMOIRE)
# -----------------------------------------------------
def generer_image(prompt_image):
    """Crée une image simulée (sans écriture disque, purement en mémoire)."""
    img = Image.new('RGB', (512, 512), color=(45, 45, 65))
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except IOError:
        font = ImageFont.load_default()

    texte = f"Simulation d'image :\n{prompt_image[:100]}..."
    d.text((20, 230), texte, fill=(255, 255, 120), font=font)

    # Conversion en mémoire (BytesIO)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

# -----------------------------------------------------
# 💬 INITIALISATION DU CHAT
# -----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.memory = []  # mémoire courte
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Bonjour ! Je suis un assistant IA léger propulsé par **GPT-2**. "
                   "Demandez-moi de générer du texte, du code, ou tapez `!image votre prompt`."
    })

# -----------------------------------------------------
# 🔄 AFFICHAGE DES MESSAGES
# -----------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------------------------
# 🧠 SAISIE UTILISATEUR
# -----------------------------------------------------
if prompt := st.chat_input("💬 Écrivez ici pour discuter..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("💭 L’IA réfléchit..."):
            réponse_finale = ""

            # 🖼️ Commande : !image
            if prompt.lower().startswith("!image"):
                prompt_image = prompt[6:].strip() or "Aucune description"
                st.info(f"🎨 Simulation d'image pour : **{prompt_image}**")

                img_bytes = generer_image(prompt_image)
                st.image(img_bytes, caption=f"Image simulée : {prompt_image}")
                réponse_finale = "Voici votre image simulée 🖼️ (CPU friendly)."

            # 🧠 Commande : !mémoire
            elif prompt.lower().startswith("!mémoire"):
                mémoire_text = "\n".join(
                    [f"- {m}" for m in st.session_state.memory[-5:]]
                ) or "Mémoire vide."
                réponse_finale = f"🧠 **Derniers sujets :**\n{mémoire_text}"

            # 💬 Réponse textuelle classique
            else:
                try:
                    contexte = " ".join(st.session_state.memory[-3:]) + " " + prompt
                    result = generator(
                        contexte,
                        max_length=180,
                        num_return_sequences=1,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True
                    )[0]['generated_text']

                    # Nettoyer la sortie
                    if result.startswith(prompt):
                        result = result[len(prompt):].strip()

                    réponse_finale = result
                    st.session_state.memory.append(prompt)
                except Exception as e:
                    réponse_finale = f"⚠️ Erreur : {e}"

        st.markdown(réponse_finale)
        st.session_state.messages.append({"role": "assistant", "content": réponse_finale})

# -----------------------------------------------------
# 🧾 PIED DE PAGE
# -----------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.9em;'>
🚀 Propulsé par GPT-2 via 🤗 Transformers | 100 % compatible Streamlit Cloud 🌐 | Interface améliorée 💬
</div>
""", unsafe_allow_html=True)    """Simule la génération d'image (placeholder)."""
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
