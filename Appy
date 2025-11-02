import streamlit as st
from transformers import pipeline
import os
from PIL import Image

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="🤖 Chatbot Léger Multi-Capacités (Open Source)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Assistant Générateur de Texte et Code Léger")
st.caption("Propulsé par le modèle **GPT-2** (Hugging Face) pour la compatibilité avec les plateformes gratuites (CPU/mémoire limitée).")
st.divider()

# --- Configuration et Chargement du Modèle (GPT-2) ---

@st.cache_resource
def load_generator():
    """Charge le pipeline du modèle GPT-2 (mis en cache)."""
    # NOTE: Remplacer "gpt2" par un modèle plus puissant nécessite plus de ressources (GPU/RAM)
    print("Chargement du modèle GPT-2...")
    generator = pipeline("text-generation", model="gpt2")
    return generator

# Charger le modèle une seule fois au démarrage
generator = load_generator()


# --- Fonctions pour la Génération d'Image (Simulée) ---
# *La vraie* génération d'image est impossible sur CPU/plateformes gratuites.
# Cette fonction simule la capacité pour maintenir la structure du chatbot.

def generer_image(prompt_image, output_filename):
    """Simule la génération d'image et crée une image placeholder simple."""
    try:
        img = Image.new('RGB', (512, 512), color = 'red')
        
        # Ajouter un texte au centre
        from PIL import ImageDraw, ImageFont
        d = ImageDraw.Draw(img)
        
        # Essayer de charger une police par défaut
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
            
        d.text((10,10), f"Image (CPU Only): {prompt_image}\n[Simulée, nécessite un GPU puissant]", fill=(255,255,0), font=font)
        img.save(output_filename)
        return True
    except Exception as e:
        st.error(f"Erreur de simulation d'image : {e}")
        return False

# --- Logique du Chatbot ---

# Initialiser l'historique de la conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Message de bienvenue initial
    st.session_state.messages.append({"role": "assistant", "content": "Bonjour ! Je suis basé sur GPT-2, capable de générer du texte et du code (utilisez la commande `!image` pour tester la capacité d'image simulée)." })

# Afficher les messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Gestion de la nouvelle entrée utilisateur
if prompt := st.chat_input("Dites bonjour ou demandez 'Écris une fonction Python...'"):
    # 1. Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Préparer la réponse du bot
    with st.chat_message("assistant"):
        with st.spinner("L'IA génère la réponse..."):
            
            # --- LOGIQUE DU CHATBOT (Capacités Légères) ---

            if prompt.lower().startswith("!image"):
                # Capacité: Génération d'Image (simulée)
                prompt_image = prompt[6:].strip()
                filename = "temp_image_output.png"
                
                st.text(f"🎨 Simulation de la génération d'image pour : {prompt_image}")
                
                if generer_image(prompt_image, filename):
                    st.image(filename, caption=f"Image Simulée pour : {prompt_image}")
                    st.success("La génération d'image a été simulée avec succès. La vraie version nécessite un GPU.")
                    réponse_finale = "Voici l'image que j'ai créée (simulation CPU)."
                    
                    # Nettoyage
                    os.remove(filename) 
                else:
                    réponse_finale = "Échec de la simulation d'image."

            else:
                # Capacité: Texte/Code/Chat (via GPT-2)
                try:
                    response_text = generator(
                        prompt,
                        max_length=250,  # Longueur maximale
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7 # Température pour un peu de créativité
                    )[0]['generated_text']

                    # Nettoyer la réponse pour enlever l'écho du prompt
                    if response_text.startswith(prompt):
                        réponse_finale = response_text[len(prompt):].strip()
                    else:
                        réponse_finale = response_text
                        
                except Exception as e:
                    st.error(f"Erreur de génération de texte : {e}")
                    réponse_finale = "Désolé, une erreur est survenue lors de la génération de la réponse textuelle."


        # 3. Affichage et sauvegarde de la réponse
        st.markdown(réponse_finale)
        st.session_state.messages.append({"role": "assistant", "content": réponse_finale})
