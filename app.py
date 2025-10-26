import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Générateur de Code IA", page_icon="🤖", layout="centered")

st.title("🤖 Générateur de Code IA")
st.markdown("**Décris le code que tu veux, choisis le langage, et laisse l’IA le créer.**")

# Clé API OpenAI
openai_api_key = st.text_input("🔑 Entre ta clé OpenAI :", type="password")

# Prompt utilisateur
prompt = st.text_area("🧠 Description du code :", placeholder="Ex: Écris un script Python qui trie une liste de nombres.")
langage = st.selectbox("💬 Langage :", ["Python", "JavaScript", "C++", "Java", "HTML/CSS"])

if st.button("🚀 Générer le code"):
    if not openai_api_key:
        st.error("⚠️ Tu dois entrer une clé API OpenAI (https://platform.openai.com/account/api-keys).")
    elif not prompt.strip():
        st.warning("📝 Écris une description avant de générer le code.")
    else:
        with st.spinner("💡 L’IA réfléchit..."):
            client = OpenAI(api_key=openai_api_key)
            reponse = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Tu es une IA experte en {langage}. Génère uniquement du code clair."},
                    {"role": "user", "content": prompt}
                ]
            )
            code = reponse.choices[0].message.content

        st.success("✅ Code généré !")
        st.code(code, language=langage.lower())
        st.download_button("⬇️ Télécharger le code", code, file_name=f"code_{langage.lower()}.txt")

st.markdown("---")
st.caption("🌐 Fait avec ❤️ en Streamlit + OpenAI")
