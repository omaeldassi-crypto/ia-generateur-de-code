import streamlit as st
import os
import torch
from PIL import Image, ImageDraw, ImageFont

# Importations LangChain et Hugging Face
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from accelerate import Accelerator # Nécessaire si vous utilisez 'accelerate' dans requirements.txt

# --- 1. Configuration Globale et Modèles Légers (Mistral 7B) ---

# NOTE : Mistral 7B est lourd, mais c'est le meilleur compromis performance/qualité sur les CPU gratuits.
LLM_MODEL_NAME = "Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.environ.get("HF_TOKEN") # Nécessaire pour les modèles lourds/populaires sur Hugging Face

# --- 2. Fonctions d'Initialisation (Cachées par Streamlit) ---

@st.cache_resource
def get_llm_pipeline_base():
    """Initialise le pipeline Mistral 7B sur CPU (Lent mais stable sur Spaces)."""
    try:
        print(f"🧠 Initialisation du LLM '{LLM_MODEL_NAME}' sur CPU...")
        
        # Le chargement d'un modèle 7B est difficile sur CPU, nous utilisons une stratégie optimisée :
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, 
            device_map="auto", # Utilise CPU ou GPU
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, # Très important pour les serveurs CPU/RAM limités
            token=HF_TOKEN
        )
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3, # Température basse pour un raisonnement d'Agent plus fiable
            top_p=0.95,
        )
        return HuggingFacePipeline(pipeline=model_pipeline)
    except Exception as e:
        st.error(f"Erreur d'initialisation du LLM Mistral : {e}. Vérifiez la mémoire de votre Space.")
        return None

@st.cache_resource
def get_agent_executor():
    """Crée l'Agent Executor LangChain avec les outils de Recherche Web et de Code."""
    llm = get_llm_pipeline_base()
    if llm is None:
        return None

    # Outil de recherche web (DuckDuckGo)
    search = DuckDuckGoSearchRun(name="DuckDuckGo_Search")
    
    # Outil pour la génération de code
    code_tool = Tool(
        name="GenerateCode",
        func=lambda x: "J'ai généré le code comme demandé. Utilise un bloc Markdown pour l'afficher.", 
        description="Utilise cet outil UNIQUEMENT pour les requêtes de génération de code (Python, Javascript, HTML, etc.)."
    )
    
    tools = [search, code_tool]

    # Prompt d'Agent ReAct (optimisé pour les modèles Instruct de Mistral)
    template = f"""Tu es un agent IA expert capable de chercher des informations sur le web et de générer du code. 
    Tu as accès aux outils suivants :
    
    {{tools}}

    Réponds aux questions en utilisant la recherche web si l'information est inconnue ou récente.
    Utilise 'GenerateCode' lorsque l'utilisateur demande explicitement d'écrire du code.

    Utilise OBLIGATOIREMENT le format ReAct suivant. Si tu n'utilises pas d'outil, va directement à Réponse Finale.

    Question: la question ici
    Pensée: Je dois raisonner si j'utilise un outil ou si je réponds directement.
    Action: Nom_de_l'outil (si applicable, ex: DuckDuckGo_Search)
    Action Input: L'entrée pour l'outil
    Observation: Le résultat de l'outil
    ... (jusqu'à la Réponse Finale)
    Réponse Finale: la réponse finale à l'utilisateur ici

    Commence!
    
    Question: {{input}}
    {{agent_scratchpad}}
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Création de l'Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Création de l'Agent Executor
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 3. Fonction d'Image (Simulation) ---

@st.cache_resource
def generer_image(prompt_image, output_filename):
    """Simule la génération d'image pour le déploiement CPU."""
    try:
        img = Image.new('RGB', (512, 512), color = '#3498db') 
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial.ttf", 25)
        except IOError:
            font = ImageFont.load_default()
            
        text = f"Image pour: {prompt_image}\n\n[Génération d'Image Simulée : GPU requis pour la vraie version]"
        d.text((10, 10), text, fill=(255, 255, 255), font=font)
        img.save(output_filename)
        return True
    except Exception as e:
        st.error(f"Erreur de simulation d'image : {e}")
        return False


# --- 4. Interface Streamlit ---

if __name__ == '__main__':
    st.set_page_config(page_title="🤖 Agent Mistral Multi-Capacités (HF Spaces)", layout="wide")
    st.title("🤖 Agent Mistral Multi-Capacités (HF Spaces)")
    st.caption(f"LLM utilisé: {LLM_MODEL_NAME} | Capacités: Code, Chat, Recherche Web, Image (Simulée)")

    agent_executor = get_agent_executor()
    
    if agent_executor is None:
        st.stop() # Arrête l'exécution si le modèle n'a pas pu charger

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Bonjour ! Je suis l'Agent Mistral. Je peux coder, chercher sur le web, et simuler la création d'images. Que puis-je faire pour vous ?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Votre requête (Ex: Fais le code Python pour trier une liste)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("L'Agent Mistral est en cours de raisonnement... (Le premier chargement est très lent sur CPU)"):
                
                if prompt.lower().startswith("!image"):
                    # Gestion de la simulation d'image
                    prompt_image = prompt[6:].strip()
                    filename = "temp_image_output.png"
                    if generer_image(prompt_image, filename): 
                        st.image(filename, caption=f"Image Simulée pour : {prompt_image}")
                        st.warning("⚠️ Ceci est une simulation. La vraie génération d'images nécessite un GPU.")
                        réponse_finale = "Voici l'image simulée que j'ai créée."
                        os.remove(filename)
                    else:
                        réponse_finale = "Désolé, la simulation d'image a échoué."
                        
                else:
                    # Gestion de l'Agent (Code, Chat, Recherche Web)
                    try:
                        resultat = agent_executor.invoke({"input": prompt})
                        réponse_finale = resultat.get('output', "Réponse non trouvée.")
                        
                    except Exception as e:
                        st.error(f"Erreur d'exécution de l'Agent : {e}")
                        réponse_finale = "Une erreur est survenue lors de l'exécution de l'Agent. Le modèle Mistral a peut-être eu du mal à suivre le format ReAct."

                st.write(réponse_finale)
                st.session_state.messages.append({"role": "assistant", "content": réponse_finale})
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    texte = f"Simulation :\n{prompt_image[:90]}..."
    draw.text((20, 230), texte, fill=(255, 255, 100), font=font)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

# ---------------------------
# HISTORIQUE DU CHAT
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.memory = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Bonjour ! Je suis un assistant IA léger basé sur distilGPT-2.\n"
                   "Utilisez '!image <description>' pour générer une image simulée."
    })

def afficher_historique():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

afficher_historique()

# ---------------------------
# ENTRÉE UTILISATEUR
# ---------------------------
if prompt := st.chat_input("Entrez votre message ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("L'IA réfléchit..."):
            réponse_finale = ""

            # Commande !image
            if prompt.lower().startswith("!image"):
                prompt_image = prompt[6:].strip() or "Aucune description"
                st.info(f"Simulation d'image pour : {prompt_image}")
                img_bytes = generer_image(prompt_image)
                st.image(img_bytes, caption=f"Image simulée : {prompt_image}")
                réponse_finale = "Voici une image simulée (version CPU)."

            # Commande !mémoire
            elif prompt.lower().startswith("!mémoire"):
                mémoire_text = "\n".join(
                    [f"- {m}" for m in st.session_state.memory[-5:]]
                ) or "Mémoire vide."
                réponse_finale = f"Derniers sujets :\n{mémoire_text}"

            # Réponse textuelle via distilGPT-2
            elif generator:
                try:
                    contexte = " ".join(st.session_state.memory[-2:]) + " " + prompt
                    result = generator(
                        contexte,
                        max_length=100,
                        num_return_sequences=1,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True
                    )[0]['generated_text']

                    if result.startswith(prompt):
                        result = result[len(prompt):].strip()

                    réponse_finale = result
                    st.session_state.memory.append(prompt)
                except Exception as e:
                    réponse_finale = f"Erreur pendant la génération : {e}"
            else:
                réponse_finale = "Le modèle n’a pas pu être chargé."

        st.session_state.messages.append({"role": "assistant", "content": réponse_finale})
        st.markdown(réponse_finale)

st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.9em;'>
Propulsé par distilGPT-2 | Compatible Streamlit Cloud | CPU uniquement
</div>
""", unsafe_allow_html=True)    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    texte = f"Simulation :\n{prompt_image[:90]}..."
    draw.text((20, 230), texte, fill=(255, 255, 100), font=font)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

# ---------------------------
# HISTORIQUE DU CHAT
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.memory = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Bonjour ! Je suis un assistant IA léger basé sur distilGPT-2.\n"
                   "Utilisez '!image <description>' pour générer une image simulée."
    })

def afficher_historique():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

afficher_historique()

# ---------------------------
# ENTRÉE UTILISATEUR
# ---------------------------
if prompt := st.chat_input("Entrez votre message ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("L'IA réfléchit..."):
            réponse_finale = ""

            # Commande !image
            if prompt.lower().startswith("!image"):
                prompt_image = prompt[6:].strip() or "Aucune description"
                st.info(f"Simulation d'image pour : {prompt_image}")
                img_bytes = generer_image(prompt_image)
                st.image(img_bytes, caption=f"Image simulée : {prompt_image}")
                réponse_finale = "Voici une image simulée (version CPU)."

            # Commande !mémoire
            elif prompt.lower().startswith("!mémoire"):
                mémoire_text = "\n".join(
                    [f"- {m}" for m in st.session_state.memory[-5:]]
                ) or "Mémoire vide."
                réponse_finale = f"Derniers sujets :\n{mémoire_text}"

            # Réponse textuelle via distilGPT-2
            elif generator:
                try:
                    contexte = " ".join(st.session_state.memory[-2:]) + " " + prompt
                    result = generator(
                        contexte,
                        max_length=100,
                        num_return_sequences=1,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True
                    )[0]['generated_text']

                    if result.startswith(prompt):
                        result = result[len(prompt):].strip()

                    réponse_finale = result
                    st.session_state.memory.append(prompt)
                except Exception as e:
                    réponse_finale = f"Erreur pendant la génération : {e}"
            else:
                réponse_finale = "Le modèle n’a pas pu être chargé."

        st.session_state.messages.append({"role": "assistant", "content": réponse_finale})
        st.markdown(réponse_finale)

st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.9em;'>
Propulsé par distilGPT-2 | Compatible Streamlit Cloud | CPU uniquement
</div>
""", unsafe_allow_html=True)# -----------------------------------------------------
def generer_image(prompt_image: str):
    """Crée une image simulée directement en mémoire."""
    img = Image.new('RGB', (512, 512), color=(40, 40, 65))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    texte = f"Simulation :\n{prompt_image[:90]}..."
    draw.text((20, 230), texte, fill=(255, 255, 100), font=font)

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

# -----------------------------------------------------
# INITIALISATION DU CHAT
# -----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.memory = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Bonjour ! Je suis un assistant IA léger basé sur distilGPT-2.\n"
                   "Utilisez '!image <description>' pour générer une image simulée."
    })

# -----------------------------------------------------
# AFFICHAGE DE L’HISTORIQUE
# -----------------------------------------------------
def afficher_historique():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

afficher_historique()

# -----------------------------------------------------
# ENTRÉE UTILISATEUR
# -----------------------------------------------------
if prompt := st.chat_input("Entrez votre message ici..."):
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Afficher immédiatement le message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Générer la réponse du bot
    with st.chat_message("assistant"):
        with st.spinner("L'IA réfléchit..."):
            réponse_finale = ""

            # Commande : !image
            if prompt.lower().startswith("!image"):
                prompt_image = prompt[6:].strip() or "Aucune description"
                st.info(f"Simulation d'image pour : {prompt_image}")
                img_bytes = generer_image(prompt_image)
                st.image(img_bytes, caption=f"Image simulée : {prompt_image}")
                réponse_finale = "Voici une image simulée (version CPU)."

            # Commande : !mémoire
            elif prompt.lower().startswith("!mémoire"):
                mémoire_text = "\n".join(
                    [f"- {m}" for m in st.session_state.memory[-5:]]
                ) or "Mémoire vide."
                réponse_finale = f"Derniers sujets :\n{mémoire_text}"

            # Réponse textuelle via distilGPT-2
            elif generator:
                try:
                    contexte = " ".join(st.session_state.memory[-2:]) + " " + prompt
                    result = generator(
                        contexte,
                        max_length=100,
                        num_return_sequences=1,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True
                    )[0]['generated_text']

                    # Nettoyer l’écho du prompt
                    if result.startswith(prompt):
                        result = result[len(prompt):].strip()

                    réponse_finale = result
                    st.session_state.memory.append(prompt)
                except Exception as e:
                    réponse_finale = f"Erreur pendant la génération : {e}"
            else:
                réponse_finale = "Le modèle n’a pas pu être chargé."

        # Ajouter la réponse à l'historique et l'afficher
        st.session_state.messages.append({"role": "assistant", "content": réponse_finale})
        st.markdown(réponse_finale)

# -----------------------------------------------------
# PIED DE PAGE
# -----------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.9em;'>
Propulsé par distilGPT-2 | Compatible Streamlit Cloud | CPU uniquement
</div>
""", unsafe_allow_html=True)# GÉNÉRATION D’IMAGE SIMULÉE
# -----------------------------------------------------
def generer_image(prompt_image: str):
    """Crée une image simulée directement en mémoire."""
    img = Image.new('RGB', (512, 512), color=(40, 40, 65))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    texte = f"Simulation :\n{prompt_image[:90]}..."
    draw.text((20, 230), texte, fill=(255, 255, 100), font=font)

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

# -----------------------------------------------------
# INITIALISATION DU CHAT
# -----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.memory = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Bonjour ! Je suis un assistant IA léger basé sur distilGPT-2.\n"
                   "Utilisez '!image <description>' pour générer une image simulée."
    })

# -----------------------------------------------------
# AFFICHAGE DE L’HISTORIQUE
# -----------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------------------------
# ENTRÉE UTILISATEUR
# -----------------------------------------------------
if prompt := st.chat_input("Entrez votre message ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("L'IA réfléchit..."):
            réponse_finale = ""

            # Commande : !image
            if prompt.lower().startswith("!image"):
                prompt_image = prompt[6:].strip() or "Aucune description"
                st.info(f"Simulation d'image pour : {prompt_image}")
                img_bytes = generer_image(prompt_image)
                st.image(img_bytes, caption=f"Image simulée : {prompt_image}")
                réponse_finale = "Voici une image simulée (version CPU)."

            # Commande : !mémoire
            elif prompt.lower().startswith("!mémoire"):
                mémoire_text = "\n".join(
                    [f"- {m}" for m in st.session_state.memory[-5:]]
                ) or "Mémoire vide."
                réponse_finale = f"Derniers sujets :\n{mémoire_text}"

            # Réponse textuelle
            elif generator:
                try:
                    contexte = " ".join(st.session_state.memory[-2:]) + " " + prompt
                    result = generator(
                        contexte,
                        max_length=100,
                        num_return_sequences=1,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True
                    )[0]['generated_text']

                    if result.startswith(prompt):
                        result = result[len(prompt):].strip()

                    réponse_finale = result
                    st.session_state.memory.append(prompt)
                except Exception as e:
                    réponse_finale = f"Erreur pendant la génération : {e}"
            else:
                réponse_finale = "Le modèle n’a pas pu être chargé."

        st.markdown(réponse_finale)
        st.session_state.messages.append({"role": "assistant", "content": réponse_finale})

# -----------------------------------------------------
# PIED DE PAGE
# -----------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.9em;'>
Propulsé par distilGPT-2 | Compatible Streamlit Cloud | CPU uniquement
</div>
""", unsafe_allow_html=True)    """Crée une image simulée (sans écriture disque, purement en mémoire)."""
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
