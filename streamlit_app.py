import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG System Demo", layout="wide")
st.title("🔎 Simple RAG System Demo")
st.markdown(
    """
    <small>
    <b>Retrieval-Augmented Generation (RAG)</b> - Adicione documentos, busque por similaridade e faça perguntas com contexto!
    </small>
    """, unsafe_allow_html=True
)

# Sidebar: Navegação
st.sidebar.title("Navegação")
page = st.sidebar.radio(
    "Escolha uma funcionalidade:",
    ("Adicionar Documento", "Buscar Documento", "RAG Chat")
)

def show_success(msg):
    st.success(msg, icon="✅")

def show_error(msg):
    st.error(msg, icon="🚫")

def add_document():
    st.header("📄 Adicionar Documento")
    with st.form("add_doc_form"):
        text = st.text_area("Conteúdo do documento (Obrigatório)", height=150, max_chars=5000, help="Limite de 5000 caracteres.")
        col1, col2, col3 = st.columns(3)
        with col1:
            contexto = st.text_input("Contexto (obrigatório)")
        with col2:
            date = st.text_input("Data (opcional)", placeholder="YYYY-MM-DD")
        with col3:
            source = st.text_input("Fonte (opcional)")
        submitted = st.form_submit_button("Adicionar")
    if submitted:
        if not text.strip():
            show_error("O conteúdo do documento não pode ser vazio.")
            return
        if not contexto.strip():
            show_error("O campo 'Contexto' é obrigatório.")
            return
        metadata = {"contexto": contexto}
        if date: metadata["date"] = date
        if source: metadata["source"] = source
        try:
            resp = requests.post(f"{API_URL}/add_document", json={"text": text, "metadata": metadata})
            if resp.status_code == 200:
                show_success(f"Documento adicionado com sucesso! ID: {resp.json().get('id')}")
            else:
                show_error(f"Erro: {resp.json().get('detail', 'Erro desconhecido')}")
        except Exception as e:
            show_error(f"Erro de conexão com API: {e}")

def search_document():
    st.header("🔍 Buscar Documento por Similaridade")
    with st.form("search_form"):
        query = st.text_input("Digite sua busca", help="Exemplo: 'Japão', 'machine learning', etc.")
        limit = st.slider("Quantidade de resultados", 1, 10, 5)
        submitted = st.form_submit_button("Buscar")
    if submitted:
        if not query.strip():
            show_error("A busca não pode ser vazia.")
            return
        try:
            resp = requests.get(f"{API_URL}/search", params={"query": query, "limit": limit})
            if resp.status_code == 200:
                results = resp.json()
                if not results:
                    show_error("Nenhum documento relevante encontrado.")
                else:
                    st.write(f"**{len(results)} resultados encontrados:**")
                    for idx, res in enumerate(results, 1):
                        with st.expander(f"Resultado {idx} (Score: {res['score']:.4f})"):
                            st.markdown(f"**Conteúdo:** {res['content']}")
                            st.markdown(f"**Metadados:** {res['metadata']}")
            else:
                show_error(f"Erro: {resp.json().get('detail', 'Erro desconhecido')}")
        except Exception as e:
            show_error(f"Erro de conexão com API: {e}")

def rag_chat():
    st.header("💬 RAG Chat")
    with st.form("chat_form"):
        question = st.text_area("Digite sua pergunta", height=80)
        max_results = st.slider("Quantidade de documentos de contexto", 1, 10, 3)
        submitted = st.form_submit_button("Perguntar")
    if submitted:
        if not question.strip():
            show_error("A pergunta não pode ser vazia.")
            return
        try:
            resp = requests.post(f"{API_URL}/chat", json={"question": question, "max_results": max_results})
            if resp.status_code == 200:
                data = resp.json()
                st.markdown(f"**Resposta:** {data['answer']}")
                st.markdown(f"**Modelo:** `{data['model_used']}` | **Tokens usados:** {data['tokens_used']}")
                if data["sources"]:
                    st.markdown("**Fontes/contexto utilizado:**")
                    for idx, src in enumerate(data["sources"], 1):
                        with st.expander(f"Fonte {idx}"):
                            st.markdown(f"**Conteúdo:** {src['content']}")
                            st.markdown(f"**Metadados:** {src['metadata']}")
                else:
                    st.info("Nenhuma fonte relevante encontrada para esta resposta.")
            else:
                show_error(f"Erro: {resp.json().get('detail', 'Erro desconhecido')}")
        except Exception as e:
            show_error(f"Erro de conexão com API: {e}")

# Roteamento de páginas
if page == "Adicionar Documento":
    add_document()
elif page == "Buscar Documento":
    search_document()
elif page == "RAG Chat":
    rag_chat()