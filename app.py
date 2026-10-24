"""Moss Scholar — interface Streamlit para agente acadêmico ReAct."""

import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from papers import get_library
from agent import make_graph

st.set_page_config(
    page_title="Moss Scholar | Pesquisa acadêmica com IA",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(Path(__file__).resolve().parent / ".env")
try:
    for key in (
        "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "OPENALEX_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT",
    ):
        if key in st.secrets and st.secrets[key]:
            os.environ[key] = str(st.secrets[key])
except Exception:
    # Em desenvolvimento local, as variáveis podem estar apenas no .env.
    pass

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
:root { --moss: #8A9A5B; --black: #000000; --ink: #E9EDD9; --muted: #A5AF98; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: 'DM Sans',sans-serif; }
.stApp { background: radial-gradient(circle at 84% 5%, #1c2513 0%, #090d09 36%, #000000 74%); color: var(--ink); }
[data-testid="stSidebar"] { background: #0b100a; border-right: 1px solid #283421; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1190px; padding-top: 2.25rem; padding-bottom: 4rem; }
h1, h2, h3 { font-family: 'Playfair Display', Georgia, serif; letter-spacing: -.02em; }
h1 { color: #EEF2DB !important; }
a { color: #B7C98B !important; }
small, .text-muted { color: var(--muted) !important; }
.hero { background: linear-gradient(113deg, #151d10 0%, #202b17 55%, #344323 100%);
   border: 1px solid #556844; border-radius: 26px; padding: 36px 42px; margin-bottom: 22px; }
.hero .eyebrow { text-transform: uppercase; letter-spacing: .2em; font-size: 12px; font-weight: 700; color: #BDC99A; }
.hero h1 { margin: 8px 0 10px; font-family:'Playfair Display',Georgia,serif; font-size: clamp(35px, 5vw, 58px); line-height:1.1; }
.hero p { color: #CBD3B7; font-size: 16px; max-width: 760px; margin: 0; line-height:1.7; }
.metric { padding: 18px 22px; border-radius: 17px; border:1px solid #394731;
    background: #11180F; margin: 2px 0 22px; }
.metric b { display: block; font-size: 25px; color: #CDDCA7; margin-bottom: 2px; }
.metric span { color: #ABB79D; font-size: 13px; }
.paper-kicker { color:#AFC284; text-transform:uppercase; letter-spacing: .1em; font-size: 11px; }
.paper-title { color:#EDF1E3; font-weight:700; font-size:17px; line-height:1.35; margin:9px 0; }
.paper-desc { color:#ADB8A6; line-height:1.65; font-size:14px; min-height:76px; }
.paper-card { padding: 20px; background: #101710; border-radius:18px; border:1px solid #35442E; min-height: 235px; margin-bottom:10px; }
.paper-meta { color:#B7C89A; font-size:12px; margin-bottom:8px; }
section[data-testid="stChatMessage"] { background:#11170e; border:1px solid #2c3926; border-radius:16px; padding:11px; }
[data-testid="stChatInput"] { background:#121a10; border-color:#596947; }
.stButton > button { border-radius:11px; border:1px solid #65774F; background-color:#1D2B18; color:#E1E9D2; }
.stButton > button:hover { background-color:#35452B; border-color:#8A9A5B; color:white; }
[data-baseweb="tab-list"] { gap: 10px; }
[data-baseweb="tab"] { color:#B1BF9B; }
hr { border-color:#35442E !important; }
.brand { font-family:'Playfair Display', Georgia, serif; color:#B5C88C; font-size:30px; font-weight:700; }
.tiny { color:#A0AA96; font-size:13px; line-height:1.6; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

papers = get_library()

with st.sidebar:
    st.markdown('<div class="brand">🌿 Moss Scholar</div>', unsafe_allow_html=True)
    st.caption("Academic research, thoughtfully done.")
    st.divider()
    st.markdown("**NAVEGAÇÃO**")
    st.markdown("Biblioteca inicial · 10 artigos")
    st.markdown("Pesquisa e comparação com IA")
    st.divider()
    st.markdown("**TECNOLOGIA**")
    st.caption("LangGraph · LangSmith · Streamlit")
    st.caption("Três ferramentas: pesquisar, extrair, comparar")
    st.divider()
    provider = os.getenv("LLM_PROVIDER", "openai").upper()
    key_value = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "")
    has_key = bool(key_value and not key_value.upper().startswith("COLOQUE_"))
    st.markdown(f"**Modelo:** {provider}")
    st.success("Chave do modelo configurada") if has_key else st.warning("Configure a chave do modelo para ativar o chat")
    st.info("OpenAlex ativo") if os.getenv("OPENALEX_API_KEY") else st.caption("Busca externa opcional: configure OPENALEX_API_KEY")
    st.caption("LangSmith ativo" if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING", "").lower() == "true" else "LangSmith: adicione a chave para registrar execuções")
    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.history = []
        st.session_state.agent_messages = []
        st.session_state.pending_question = None
        st.rerun()

st.markdown("""
<div class="hero">
  <span class="eyebrow">Research assistant • ReAct agent</span>
  <h1>Pesquisa com mais<br/>clareza e contexto.</h1>
  <p>Explore artigos científicos, encontre metodologias e compare estudos com um agente de IA que seleciona as ferramentas certas para cada pergunta.</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="metric"><b>10</b><span>Artigos iniciais do arXiv</span></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric"><b>03</b><span>Ferramentas especializadas</span></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric"><b>ReAct</b><span>Decisão e execução de ferramentas</span></div>', unsafe_allow_html=True)

chat_tab, library_tab, about_tab = st.tabs(["✦ Assistente", "◉ Biblioteca", "◎ Sobre o projeto"])


def show_trace(events: list[dict]):
    if not events:
        return
    with st.expander(f"Ferramentas utilizadas ({len(events)} etapa(s))", expanded=False):
        for i, event in enumerate(events, start=1):
            st.markdown(f"**{i}. `{event['name']}`**")
            with st.container(border=True):
                st.code(event["result"][:1100] + ("…" if len(event["result"]) > 1100 else ""), language="json")


def content_as_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(item.get("text", "") for item in value if isinstance(item, dict) and item.get("type") == "text")
    return str(value)


with chat_tab:
    st.subheader("Seu assistente de pesquisa")
    st.caption("Escreva na barra fixa no rodapé ou escolha uma das sugestões. A barra continua disponível após a resposta.")

    prompts = [
        "Encontre artigos introdutórios sobre detecção de fake news.",
        "Compare os estudos 2309.11052 e 2411.12703 sobre métodos de classificação.",
        "Quais artigos usam imagens além de texto? Compare dois deles.",
    ]
    cols = st.columns(3)
    for i, prompt in enumerate(prompts):
        with cols[i]:
            if st.button(["Pesquisa inicial", "Comparar metodologias", "Texto + imagem"][i],
                         key=f"p{i}", use_container_width=True):
                st.session_state.pending_question = prompt
                st.rerun()

    # A pergunta é processada ANTES da exibição do histórico. Isso impede que
    # uma resposta longa apareça abaixo da barra de entrada.
    question = st.session_state.pending_question
    if question:
        st.session_state.pending_question = None  # Evita executar novamente em reruns.
        st.session_state.history.append({"role": "user", "content": question})
        if not has_key:
            reply = (
                "Para ativar o agente, configure `LLM_API_KEY` em **Settings → Secrets** "
                "do Streamlit Cloud e `LLM_PROVIDER = \"openrouter\"` se usar OpenRouter. "
                "A biblioteca local continua disponível sem essa chave."
            )
            trace = []
        else:
            with st.spinner("Pesquisando e consultando as ferramentas do agente..."):
                try:
                    previous = st.session_state.agent_messages
                    graph = make_graph()
                    response = graph.invoke(
                        {"messages": previous + [HumanMessage(content=question)]},
                        config={"recursion_limit": 24, "run_name": "MossScholar-ReAct"},
                    )
                    current = response["messages"]
                    new_messages = current[len(previous) + 1:]
                    reply = next(
                        (content_as_text(msg.content) for msg in reversed(current)
                         if isinstance(msg, AIMessage) and not msg.tool_calls),
                        "O modelo terminou sem produzir uma resposta final. Tente novamente.",
                    )
                    trace = [
                        {"name": m.name or "ferramenta", "result": content_as_text(m.content)}
                        for m in new_messages if isinstance(m, ToolMessage)
                    ]
                    st.session_state.agent_messages = current
                except Exception as exc:
                    # Não exibir o conteúdo bruto da exceção: respostas de APIs
                    # podem conter detalhes internos ou segredos de configuração.
                    category = type(exc).__name__
                    message = str(exc).lower()
                    if "401" in message or "invalid api key" in message:
                        tip = "Confira a API Key em Settings → Secrets."
                    elif "429" in message or "rate limit" in message:
                        tip = "O OpenRouter atingiu o limite de requisições. Tente mais tarde."
                    elif "tool" in message and ("support" in message or "unsupported" in message):
                        tip = "O modelo selecionado não aceita chamadas de ferramentas. Selecione outro modelo com tool calling."
                    elif "recursion" in message:
                        tip = "O agente atingiu seu limite de etapas. Faça uma pergunta mais específica."
                    else:
                        tip = "Confira o modelo no OpenRouter e os registros do app no Streamlit Cloud."
                    reply = f"⚠️ Não foi possível concluir esta pesquisa ({category}). {tip}"
                    trace = []

        st.session_state.history.append({"role": "assistant", "content": reply, "trace": trace})

    for item in st.session_state.history:
        with st.chat_message(item["role"], avatar="🌿" if item["role"] == "assistant" else "👤"):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                trace = item.get("trace", [])
                show_trace(trace)
                if trace == [] and "⚠️" not in item["content"] and "Para ativar" not in item["content"]:
                    st.caption("Este modelo respondeu sem executar ferramentas. Para consultas verificadas, escolha um modelo com suporte a tool calling.")

with library_tab:
    st.subheader("Biblioteca de artigos")
    st.caption("Seleção inicial de 10 trabalhos reais. Os cartões apresentam sínteses curadas e links para o arXiv; o aplicativo não armazena nem analisa o PDF completo.")
    query = st.text_input("Filtrar por título, assunto ou técnica", placeholder="Ex.: português, BERT, multimodal, datasets")
    if query.strip():
        from papers import search_library
        shown = search_library(query, 10)
    else:
        shown = papers
    st.caption(f"{len(shown)} artigo(s) encontrado(s)")
    for start in range(0, len(shown), 2):
        columns = st.columns(2)
        for index, paper in enumerate(shown[start:start + 2]):
            with columns[index]:
                title = html.escape(paper["title"])
                topic = html.escape(paper["topic"])
                summary = html.escape(paper["summary_pt"])
                st.markdown(f"""
                <div class="paper-card">
                    <div class="paper-kicker">{topic}</div>
                    <div class="paper-title">{title}</div>
                    <div class="paper-meta">{paper['year']} · arXiv:{paper['id']}</div>
                    <div class="paper-desc">{summary}</div>
                </div>
                """, unsafe_allow_html=True)
                st.link_button("Ver artigo ↗", paper["url"], use_container_width=True)
                with st.expander("Autores e métodos"):
                    st.write(", ".join(paper["authors"]))
                    st.caption("Métodos/temas: " + ", ".join(paper["methods"]))

with about_tab:
    st.subheader("Como o agente funciona")
    st.markdown("""
**1. Reasoning:** o modelo interpreta a pergunta e escolhe a ferramenta apropriada.

**2. Acting:** o LangGraph encaminha a chamada à ferramenta e recebe a observação.

**3. Observation:** o modelo examina os metadados retornados, decide se precisa de outra consulta ou escreve a resposta final.

**As três ferramentas:** `search_papers` procura artigos, `extract_information` recupera informações de um artigo e `compare_studies` organiza a comparação de 2 a 5 estudos.

**LangSmith:** habilitado por variáveis de ambiente, registra o fluxo do grafo e as chamadas para depuração e avaliação. O painel *Ferramentas utilizadas* mostra apenas nomes e resultados, não pensamentos internos do modelo.

**Limitação:** a análise usa metadados, resumos consultados e sínteses da biblioteca. Para uma revisão de literatura rigorosa, leia os artigos completos e confira as referências.
""")
    st.code("""START → agent (LLM) → tools_condition
                           ├─ tools (search / extract / compare) ──┐
                           └─ END (resposta)                      │
                       agent ◀─────────────────────────────────────┘""", language="text")

st.divider()
st.markdown('<div class="tiny">Moss Scholar · Streamlit × LangGraph × LangSmith · Identidade: Black #000000 + Moss Green #8A9A5B</div>', unsafe_allow_html=True)

# Entrada de chat FIXA (no corpo principal, não dentro de st.tabs()).
submitted = st.chat_input("Pesquise artigos ou compare estudos...", key="moss_fixed_chat")
if submitted and submitted.strip():
    st.session_state.pending_question = submitted.strip()
    st.rerun()
