"""Moss Scholar — multilingual Streamlit interface for the existing ReAct research agent.

Replace the existing app.py with this file. Requires the same papers.py, agent.py,
Python dependencies, and environment/secrets as the original application.
"""

import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Configurações devem ser aplicadas ANTES de importar os módulos do agente.
# Isso evita que modelos sejam inicializados com a chave ou endpoint errados.
st.set_page_config(
    page_title="Moss Scholar | Academic Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(Path(__file__).resolve().parent / ".env")

CONFIG_KEYS = (
    "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "OPENAI_API_KEY",
    "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENAI_BASE_URL",
    "OPENALEX_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
)

# Usar as Secrets do Streamlit Cloud com prioridade sobre o .env local.
# Ausência de secrets não deve impedir execução local.
secrets_status = "ok"
try:
    for config_key in CONFIG_KEYS:
        secret = st.secrets.get(config_key)
        if secret is not None and str(secret).strip():
            os.environ[config_key] = str(secret).strip()
except FileNotFoundError:
    secrets_status = "not_found"
except Exception:
    # Não mostrar exceções brutas, que podem conter conteúdo sensível.
    secrets_status = "unreadable"


def usable_key(value: str | None) -> bool:
    if not value:
        return False
    candidate = value.strip()
    if not candidate:
        return False
    placeholders = ("COLE_", "COLOQUE_", "YOUR_", "SUA_", "SK-OR-V1-EXEMPLO")
    return not candidate.upper().startswith(placeholders)


raw_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
shared_key = os.getenv("LLM_API_KEY", "").strip()
router_key = os.getenv("OPENROUTER_API_KEY", "").strip()
openai_key = os.getenv("OPENAI_API_KEY", "").strip()

# Reconhecer chaves sk-or-v1 e situações onde só OPENROUTER_API_KEY
# foi configurada; evita encaminhar credenciais do OpenRouter à OpenAI.
looks_like_router_key = shared_key.startswith("sk-or-") or openai_key.startswith("sk-or-")
router_only = usable_key(router_key) and not usable_key(shared_key) and not usable_key(openai_key)
provider_adjusted = False
if raw_provider not in ("openai", "openrouter"):
    provider = "openrouter" if (looks_like_router_key or router_only or usable_key(router_key)) else "openai"
elif raw_provider == "openai" and (looks_like_router_key or router_only):
    provider = "openrouter"
    provider_adjusted = True
else:
    provider = raw_provider

api_source = "não encontrada"
if provider == "openrouter":
    for name, value in (("OPENROUTER_API_KEY", router_key), ("LLM_API_KEY", shared_key), ("OPENAI_API_KEY", openai_key)):
        if usable_key(value):
            selected_key = value
            api_source = name
            break
    else:
        selected_key = ""
    if selected_key:
        os.environ["LLM_API_KEY"] = selected_key
        os.environ["OPENROUTER_API_KEY"] = selected_key
        # Compatibilidade com agentes que criam ChatOpenAI sem api_key explícita.
        os.environ["OPENAI_API_KEY"] = selected_key
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    os.environ["OPENROUTER_BASE_URL"] = base_url
    os.environ["OPENAI_BASE_URL"] = base_url
    if not os.getenv("LLM_MODEL", "").strip():
        os.environ["LLM_MODEL"] = "openrouter/free"
else:
    selected_key = shared_key if usable_key(shared_key) else (openai_key if usable_key(openai_key) else "")
    api_source = "LLM_API_KEY" if usable_key(shared_key) else ("OPENAI_API_KEY" if usable_key(openai_key) else "não encontrada")
    if selected_key:
        os.environ["LLM_API_KEY"] = selected_key
        os.environ["OPENAI_API_KEY"] = selected_key
    # Se o provedor é OpenAI, não conservar endpoint OpenRouter de sessões anteriores.
    if os.getenv("OPENAI_BASE_URL", "").startswith("https://openrouter.ai/"):
        os.environ.pop("OPENAI_BASE_URL", None)
    if not os.getenv("LLM_MODEL", "").strip():
        os.environ["LLM_MODEL"] = "gpt-4o-mini"

os.environ["LLM_PROVIDER"] = provider

# A importação tardia do agente é proposital: ele só será inicializado
# quando já tiver acesso aos Secrets e à configuração normalizada.
from papers import get_library


def make_configured_graph():
    from agent import make_graph
    return make_graph()

LANGUAGES = {"pt": "🇧🇷 Português", "en": "🇺🇸 English", "es": "🇪🇸 Español"}

I18N = {
    "pt": {
        "subtitle": "INTELIGÊNCIA ACADÊMICA", "language": "Idioma da interface",
        "workspace": "ESPAÇO DE PESQUISA", "side_nav": "AMBIENTE",
        "nav_assistant": "Agente de pesquisa", "nav_library": "Biblioteca de artigos",
        "nav_about": "Arquitetura e métodos", "side_engine": "SISTEMA",
        "model": "Provedor de IA", "configured": "Chave encontrada (ainda não validada)",
        "not_configured": "Configure a chave do modelo", "openalex_yes": "OpenAlex configurado",
        "openalex_no": "OpenAlex: chave opcional não configurada",
        "langsmith_yes": "Rastreamento LangSmith ativo", "langsmith_no": "LangSmith não ativado",
        "reset": "Nova pesquisa · limpar conversa", "side_foot": "FEITO PARA INVESTIGAR MAIS A FUNDO",
        "eyebrow": "SEU WORKSPACE DE PESQUISA COM IA", "hero_title": "Ideias melhores começam",
        "hero_emphasis": "com perguntas melhores.",
        "hero_desc": "Descubra artigos, extraia evidências e compare pesquisas em um único ambiente. Um agente ReAct seleciona as ferramentas adequadas para cada tarefa.",
        "hero_status": "AGENTE DE PESQUISA", "hero_status_value": "Pronto para investigar",
        "node1": "Descobrir", "node2": "Interpretar", "node3": "Comparar",
        "graph_tag": "FLUXO DE INVESTIGAÇÃO", "graph_source": "FONTES CIENTÍFICAS",
        "metric_papers": "ARTIGOS NA BIBLIOTECA", "metric_tools": "FERRAMENTAS DO AGENTE",
        "metric_method": "ARQUITETURA DE RACIOCÍNIO",
        "tab_assistant": "✦  Pesquisa", "tab_library": "▦  Biblioteca", "tab_about": "◇  Como funciona",
        "section_eyebrow": "01 / EXPLORAÇÃO", "section_title": "Centro de pesquisa",
        "section_desc": "Escolha um ponto de partida ou escreva uma pergunta no campo fixo abaixo. Suas investigações permanecem nesta sessão.",
        "quick_eyebrow": "COMECE POR UMA AÇÃO", "prompt1_title": "Explorar literatura",
        "prompt1_desc": "Encontre trabalhos relevantes por tema e objetivo.",
        "prompt2_title": "Comparar estudos", "prompt2_desc": "Analise diferenças entre métodos e resultados.",
        "prompt3_title": "Investigar multimodalidade", "prompt3_desc": "Explore artigos que combinam texto e imagem.",
        "prompt1": "Encontre artigos introdutórios sobre detecção de fake news.",
        "prompt2": "Compare os estudos 2309.11052 e 2411.12703 sobre métodos de classificação.",
        "prompt3": "Quais artigos usam imagens além de texto? Compare dois deles.",
        "session_label": "SESSÃO DE PESQUISA", "session_empty": "Suas pesquisas aparecerão aqui.",
        "assistant_label": "MOSS / ANÁLISE", "you_label": "SUA PERGUNTA",
        "trace": "Etapas e ferramentas utilizadas", "trace_step": "etapa(s)",
        "working": "Consultando fontes e executando as ferramentas do agente...",
        "no_key": "Para ativar o agente, configure `LLM_API_KEY` em **Settings → Secrets** do Streamlit Cloud e `LLM_PROVIDER = \"openrouter\" se usar OpenRouter. A biblioteca local continua disponível sem essa chave.",
        "no_answer": "O modelo terminou sem produzir uma resposta final. Tente novamente.",
        "failure": "Não foi possível concluir esta pesquisa",
        "error_401": "Confira a API Key em Settings → Secrets.",
        "error_429": "O modelo atingiu o limite de requisições. Tente novamente mais tarde.",
        "error_tool": "O modelo selecionado não aceita chamadas de ferramentas. Selecione um modelo com tool calling.",
        "error_recursion": "O agente atingiu o limite de etapas. Faça uma pergunta mais específica.",
        "error_generic": "Confira a configuração do modelo e os registros do aplicativo.",
        "no_tools": "Esta resposta não utilizou ferramentas. Para consultas verificadas, use um modelo compatível com tool calling.",
        "library_kicker": "02 / ACERVO", "library_title": "Biblioteca científica",
        "library_desc": "Explore uma seleção inicial de artigos com metadados e links para os estudos originais. O aplicativo não lê automaticamente os PDFs completos.",
        "library_filter": "Buscar na biblioteca", "library_placeholder": "Título, tema, BERT, multimodal...",
        "library_found": "artigo(s) encontrado(s)", "view_paper": "Abrir artigo no arXiv ↗",
        "authors_methods": "Autores e métodos", "methods": "Métodos / temas",
        "summary_note": "Os resumos dos artigos são exibidos no idioma disponível na base; a tradução automática não é realizada.",
        "about_kicker": "03 / ARQUITETURA", "about_title": "Como o Moss Scholar pensa e age",
        "about_desc": "Uma interface de pesquisa construída sobre um agente ReAct. A camada visual organiza o trabalho, enquanto o fluxo existente continua responsável pela busca, extração e comparação.",
        "phase1": "01 / INTERPRETAR", "phase1_desc": "O modelo interpreta a pergunta e seleciona uma ação adequada.",
        "phase2": "02 / EXECUTAR", "phase2_desc": "O LangGraph encaminha a chamada à ferramenta e recebe o resultado.",
        "phase3": "03 / COMPOR", "phase3_desc": "O agente examina as evidências e constrói uma resposta ou realiza outra consulta.",
        "tool_title": "FERRAMENTAS DISPONÍVEIS", "tool_search": "Buscar artigos científicos.",
        "tool_extract": "Extrair informações de um artigo.",
        "tool_compare": "Organizar a comparação de 2 a 5 estudos.",
        "observability": "OBSERVABILIDADE", "observability_desc": "O LangSmith pode registrar o fluxo do grafo quando configurado. O painel de etapas mostra apenas ferramentas e resultados, não o raciocínio interno do modelo.",
        "limitation_title": "ESCOPO E LIMITAÇÕES", "limitation_desc": "O agente utiliza metadados, resumos consultados e sínteses da biblioteca. Para uma revisão rigorosa, leia os artigos na íntegra e confira as referências.",
        "footer": "Moss Scholar / Ambiente de pesquisa acadêmica", "footer_right": "Projetado para pesquisa, não para conversa genérica",
        "chat_input": "Pergunte sobre artigos, métodos ou comparações...",
    },
    "en": {
        "subtitle": "ACADEMIC INTELLIGENCE", "language": "Interface language",
        "workspace": "RESEARCH WORKSPACE", "side_nav": "WORKSPACE",
        "nav_assistant": "Research agent", "nav_library": "Paper library",
        "nav_about": "Architecture & methods", "side_engine": "SYSTEM",
        "model": "AI provider", "configured": "API key found (not yet validated)",
        "not_configured": "Configure your model API key", "openalex_yes": "OpenAlex configured",
        "openalex_no": "OpenAlex: optional key not configured",
        "langsmith_yes": "LangSmith tracing enabled", "langsmith_no": "LangSmith not enabled",
        "reset": "New research · clear session", "side_foot": "BUILT FOR DEEPER INVESTIGATION",
        "eyebrow": "YOUR AI RESEARCH WORKSPACE", "hero_title": "Better ideas begin",
        "hero_emphasis": "with better questions.",
        "hero_desc": "Discover papers, extract evidence, and compare research in one focused workspace. A ReAct agent selects the right tools for each task.",
        "hero_status": "RESEARCH AGENT", "hero_status_value": "Ready to investigate",
        "node1": "Discover", "node2": "Interpret", "node3": "Compare",
        "graph_tag": "RESEARCH WORKFLOW", "graph_source": "SCIENTIFIC SOURCES",
        "metric_papers": "PAPERS IN LIBRARY", "metric_tools": "AGENT TOOLS",
        "metric_method": "REASONING ARCHITECTURE",
        "tab_assistant": "✦  Research", "tab_library": "▦  Library", "tab_about": "◇  How it works",
        "section_eyebrow": "01 / EXPLORATION", "section_title": "Research center",
        "section_desc": "Choose a starting point or type a question into the fixed input below. Your research stays in this session.",
        "quick_eyebrow": "START WITH AN ACTION", "prompt1_title": "Explore the literature",
        "prompt1_desc": "Find relevant studies by topic and purpose.",
        "prompt2_title": "Compare studies", "prompt2_desc": "Analyze methods and results side by side.",
        "prompt3_title": "Explore multimodal AI", "prompt3_desc": "Find papers combining text and images.",
        "prompt1": "Find introductory papers about fake news detection.",
        "prompt2": "Compare papers 2309.11052 and 2411.12703 on classification methods.",
        "prompt3": "Which papers use images in addition to text? Compare two of them.",
        "session_label": "RESEARCH SESSION", "session_empty": "Your research results will appear here.",
        "assistant_label": "MOSS / ANALYSIS", "you_label": "YOUR QUESTION",
        "trace": "Tool calls and execution steps", "trace_step": "step(s)",
        "working": "Checking sources and running agent tools...",
        "no_key": "To activate the agent, configure `LLM_API_KEY` under Streamlit Cloud **Settings → Secrets**, and set `LLM_PROVIDER = \"openrouter\" if you're using OpenRouter. The local library remains available without a key.",
        "no_answer": "The model finished without a final answer. Please try again.",
        "failure": "Unable to complete this research",
        "error_401": "Check your API key in Settings → Secrets.",
        "error_429": "The model hit its rate limit. Please try again later.",
        "error_tool": "The selected model does not support tool calls. Choose a model with tool calling.",
        "error_recursion": "The agent reached its step limit. Ask a more specific question.",
        "error_generic": "Check the model configuration and application logs.",
        "no_tools": "This response did not use tools. For verifiable research, choose a model with tool calling support.",
        "library_kicker": "02 / COLLECTION", "library_title": "Research library",
        "library_desc": "Browse an initial selection of papers, metadata, and links to original studies. The app does not automatically read full PDFs.",
        "library_filter": "Search the library", "library_placeholder": "Title, subject, BERT, multimodal...",
        "library_found": "paper(s) found", "view_paper": "Open paper on arXiv ↗",
        "authors_methods": "Authors & methods", "methods": "Methods / topics",
        "summary_note": "Paper summaries appear in the language available in the dataset; automatic translation is not performed.",
        "about_kicker": "03 / ARCHITECTURE", "about_title": "How Moss Scholar thinks and acts",
        "about_desc": "A research interface built around a ReAct agent. The visual layer organizes the work while the existing agent flow continues to handle search, extraction, and comparison.",
        "phase1": "01 / REASON", "phase1_desc": "The model interprets the question and chooses an appropriate action.",
        "phase2": "02 / ACT", "phase2_desc": "LangGraph dispatches the tool call and receives its result.",
        "phase3": "03 / SYNTHESIZE", "phase3_desc": "The agent evaluates the evidence and writes a response or continues searching.",
        "tool_title": "AVAILABLE TOOLS", "tool_search": "Search for academic papers.",
        "tool_extract": "Extract information from a paper.",
        "tool_compare": "Organize comparisons of 2 to 5 studies.",
        "observability": "OBSERVABILITY", "observability_desc": "LangSmith can record graph execution when configured. The execution panel shows tool names and outputs only, not the model's private reasoning.",
        "limitation_title": "SCOPE & LIMITATIONS", "limitation_desc": "The agent uses metadata, retrieved abstracts, and curated library summaries. For rigorous reviews, read the complete papers and verify references.",
        "footer": "Moss Scholar / Academic research workspace", "footer_right": "Designed for research, not generic chat",
        "chat_input": "Ask about papers, methods, or comparisons...",
    },
    "es": {
        "subtitle": "INTELIGENCIA ACADÉMICA", "language": "Idioma de la interfaz",
        "workspace": "ESPACIO DE INVESTIGACIÓN", "side_nav": "ENTORNO",
        "nav_assistant": "Agente de investigación", "nav_library": "Biblioteca de artículos",
        "nav_about": "Arquitectura y métodos", "side_engine": "SISTEMA",
        "model": "Proveedor de IA", "configured": "Clave encontrada (todavía no validada)",
        "not_configured": "Configura la clave del modelo", "openalex_yes": "OpenAlex configurado",
        "openalex_no": "OpenAlex: clave opcional no configurada",
        "langsmith_yes": "Seguimiento LangSmith activo", "langsmith_no": "LangSmith no activado",
        "reset": "Nueva investigación · borrar conversación", "side_foot": "CREADO PARA INVESTIGAR A FONDO",
        "eyebrow": "TU ESPACIO DE INVESTIGACIÓN CON IA", "hero_title": "Las mejores ideas comienzan",
        "hero_emphasis": "con mejores preguntas.",
        "hero_desc": "Descubre artículos, extrae evidencias y compara investigaciones en un solo entorno. Un agente ReAct elige las herramientas adecuadas para cada tarea.",
        "hero_status": "AGENTE DE INVESTIGACIÓN", "hero_status_value": "Listo para investigar",
        "node1": "Descubrir", "node2": "Interpretar", "node3": "Comparar",
        "graph_tag": "FLUJO DE INVESTIGACIÓN", "graph_source": "FUENTES CIENTÍFICAS",
        "metric_papers": "ARTÍCULOS EN BIBLIOTECA", "metric_tools": "HERRAMIENTAS DEL AGENTE",
        "metric_method": "ARQUITECTURA DE RAZONAMIENTO",
        "tab_assistant": "✦  Investigar", "tab_library": "▦  Biblioteca", "tab_about": "◇  Cómo funciona",
        "section_eyebrow": "01 / EXPLORACIÓN", "section_title": "Centro de investigación",
        "section_desc": "Elige un punto de partida o escribe una pregunta en el campo fijo de abajo. Tu investigación permanece en esta sesión.",
        "quick_eyebrow": "EMPIEZA CON UNA ACCIÓN", "prompt1_title": "Explorar la literatura",
        "prompt1_desc": "Encuentra estudios relevantes por tema y objetivo.",
        "prompt2_title": "Comparar estudios", "prompt2_desc": "Analiza diferencias entre métodos y resultados.",
        "prompt3_title": "Investigar lo multimodal", "prompt3_desc": "Explora artículos que combinan texto e imágenes.",
        "prompt1": "Encuentra artículos introductorios sobre detección de noticias falsas.",
        "prompt2": "Compara los estudios 2309.11052 y 2411.12703 sobre métodos de clasificación.",
        "prompt3": "¿Qué artículos usan imágenes además de texto? Compara dos de ellos.",
        "session_label": "SESIÓN DE INVESTIGACIÓN", "session_empty": "Los resultados de tu investigación aparecerán aquí.",
        "assistant_label": "MOSS / ANÁLISIS", "you_label": "TU PREGUNTA",
        "trace": "Herramientas y pasos ejecutados", "trace_step": "paso(s)",
        "working": "Consultando fuentes y ejecutando las herramientas del agente...",
        "no_key": "Para activar el agente, configura `LLM_API_KEY` en **Settings → Secrets** de Streamlit Cloud y `LLM_PROVIDER = \"openrouter\" si utilizas OpenRouter. La biblioteca local sigue disponible sin clave.",
        "no_answer": "El modelo terminó sin generar una respuesta final. Inténtalo de nuevo.",
        "failure": "No se pudo completar esta investigación",
        "error_401": "Comprueba tu clave API en Settings → Secrets.",
        "error_429": "El modelo alcanzó el límite de solicitudes. Vuelve a intentarlo más tarde.",
        "error_tool": "El modelo seleccionado no admite llamadas a herramientas. Elige uno con tool calling.",
        "error_recursion": "El agente alcanzó su límite de pasos. Haz una pregunta más específica.",
        "error_generic": "Comprueba la configuración del modelo y los registros de la aplicación.",
        "no_tools": "Esta respuesta no utilizó herramientas. Para consultas verificables, elige un modelo compatible con tool calling.",
        "library_kicker": "02 / COLECCIÓN", "library_title": "Biblioteca científica",
        "library_desc": "Explora una selección inicial de artículos con metadatos y enlaces a los estudios originales. La aplicación no lee automáticamente los PDF completos.",
        "library_filter": "Buscar en la biblioteca", "library_placeholder": "Título, tema, BERT, multimodal...",
        "library_found": "artículo(s) encontrado(s)", "view_paper": "Abrir artículo en arXiv ↗",
        "authors_methods": "Autores y métodos", "methods": "Métodos / temas",
        "summary_note": "Los resúmenes aparecen en el idioma disponible en la base de datos; no se traducen automáticamente.",
        "about_kicker": "03 / ARQUITECTURA", "about_title": "Cómo piensa y actúa Moss Scholar",
        "about_desc": "Una interfaz de investigación basada en un agente ReAct. La capa visual organiza el trabajo y el flujo actual sigue encargado de buscar, extraer y comparar.",
        "phase1": "01 / INTERPRETAR", "phase1_desc": "El modelo interpreta la pregunta y selecciona una acción adecuada.",
        "phase2": "02 / EJECUTAR", "phase2_desc": "LangGraph envía la llamada a la herramienta y recibe el resultado.",
        "phase3": "03 / SINTETIZAR", "phase3_desc": "El agente evalúa la evidencia y redacta una respuesta o continúa buscando.",
        "tool_title": "HERRAMIENTAS DISPONIBLES", "tool_search": "Buscar artículos científicos.",
        "tool_extract": "Extraer información de un artículo.",
        "tool_compare": "Organizar comparaciones de 2 a 5 estudios.",
        "observability": "OBSERVABILIDAD", "observability_desc": "LangSmith puede registrar el flujo del grafo cuando está configurado. El panel de ejecución muestra solo nombres y resultados de herramientas, no el razonamiento privado del modelo.",
        "limitation_title": "ALCANCE Y LIMITACIONES", "limitation_desc": "El agente utiliza metadatos, resúmenes consultados y síntesis de la biblioteca. Para revisiones rigurosas, lee los artículos completos y verifica las referencias.",
        "footer": "Moss Scholar / Entorno de investigación académica", "footer_right": "Diseñado para investigar, no para chatear sin rumbo",
        "chat_input": "Pregunta sobre artículos, métodos o comparaciones...",
    },
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --bg: #080b0b; --surface:#101716; --surface2:#131d1b; --line:#273633;
  --line-soft:#1b2826; --moss:#b5edb8; --moss-deep:#7fc89e;
  --text:#eff5f0; --muted:#98a9a0; --mono:'JetBrains Mono',monospace;
}
html, body, .stApp, button, input, textarea, [data-testid="stAppViewContainer"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse 52% 35% at 72% 0%,rgba(79,137,107,.15),transparent 85%),radial-gradient(ellipse 44% 31% at 8% 45%,rgba(48,87,77,.08),transparent 85%),var(--bg);color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] { visibility:hidden; }
[data-testid="stSidebar"] { background:#0b1110; border-right:1px solid #23302b; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.7rem; }
.block-container { max-width:1330px; padding-top:2.4rem; padding-bottom:7rem; }
[data-testid="stSidebar"] .stSelectbox label { font-size:.73rem; color:#a6b6ac; letter-spacing:.04em; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:#15211d; border-color:#34463a; color:#e3f0e6; border-radius:11px; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { color:#c6ddc9; }
[data-testid="stSidebar"] .stButton button { font-weight:600; font-size:.78rem; }
.stMarkdown p { color:var(--text); }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em; }
hr { border-color:#27332e !important; }
a { color:#b8ecc2; }
/* Sidebar identity */
.moss-brand { display:flex; align-items:center; gap:13px; margin:0 0 18px; }
.moss-mark { flex:none;display:grid;place-items:center;width:46px;height:46px;background:linear-gradient(145deg,#c6f9c8,#82d6b1);border-radius:14px;color:#0c221a;font-size:25px;box-shadow:0 8px 32px #4bb98722; }
.moss-wordmark { font-family:'Space Grotesk',sans-serif;color:#eef8ed;font-size:19px;line-height:1.05; font-weight:700;letter-spacing:-.05em; }
.moss-wordmark small { display:block; font:500 9px var(--mono);color:#8fa899;letter-spacing:.11em;margin-top:8px; }
.side-kicker { font:600 10px var(--mono); letter-spacing:.15em; color:#819589; margin:19px 0 12px; }
.side-link { display:flex;align-items:center;gap:12px;padding:11px 13px;border-radius:11px;color:#c3d4c9;background:#101a16;border:1px solid #1c2a23;margin-bottom:7px;font-size:13px; }
.side-link b { font:600 11px var(--mono);color:#8dccaa; }
.status-card {background:linear-gradient(150deg,#132019,#0e1715);border:1px solid #2a4236;border-radius:16px;padding:16px;margin:14px 0 18px;}
.status-top { font:600 10px var(--mono); color:#89af96;letter-spacing:.12em; }
.status-model { font:600 18px 'Space Grotesk',sans-serif;color:#e5f4e7;margin:9px 0;overflow-wrap:anywhere; }
.status-row {color:#a6c4af;font-size:12px;line-height:1.8;}
.status-dot {display:inline-block;width:7px;height:7px;border-radius:50%;background:#8de5ad;margin-right:8px;box-shadow:0 0 11px #76e2a8aa;}
.status-dot.off { background:#d2ae7d;box-shadow:none; }
.side-footer {color:#657d71;font:500 9px var(--mono);line-height:1.8;letter-spacing:.1em;margin-top:22px;}
/* Persistent language selector, visible even with collapsed sidebar */
.moss-main-brand {padding:17px 0 9px;color:#d9f7df;font:700 15px 'Space Grotesk',sans-serif;letter-spacing:.09em;}
.moss-main-brand span {color:#83aa92;font:500 10px var(--mono);letter-spacing:.09em;}
[data-testid="stMain"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background:#14231b; border:1px solid #4b6a56; border-radius:11px; color:#eff8f1;
}
[data-testid="stMain"] [data-testid="stSelectbox"] label {color:#bdd8c5;font:600 12px 'DM Sans',sans-serif;}
/* Top shell */
.topline {display:flex;justify-content:space-between;align-items:center;gap:10px;padding:0 0 22px;margin-bottom:28px;border-bottom:1px solid #24332c;flex-wrap:wrap;}
.topline-left {font:600 10px var(--mono); color:#9aafa3; letter-spacing:.16em;}
.topline-left strong { color:#c2ecca;font-weight:600; }
.topline-right {font:500 10px var(--mono);color:#94bba4;letter-spacing:.06em;display:flex;align-items:center;gap:9px;}
.topline-right i {display:block;width:7px;height:7px;border-radius:50%;background:#8bdba8;box-shadow:0 0 12px #78d49b88;}
/* Hero */
.hero-shell { position:relative;display:grid;grid-template-columns:1.05fr .95fr;min-height:357px;border:1px solid #355346;border-radius:25px;background:radial-gradient(circle at 82% 45%,#243a2f 0%,#15221d 40%,#0e1715 82%);overflow:hidden;margin-bottom:17px; }
.hero-shell::before {content:'';pointer-events:none;position:absolute;inset:0;background-image:linear-gradient(#aec6af08 1px,transparent 1px),linear-gradient(90deg,#aec6af08 1px,transparent 1px);background-size:34px 34px;mask-image:linear-gradient(to right,transparent 12%,#000 90%);}
.hero-copy {position:relative;z-index:1;padding:48px 22px 42px 46px;}
.hero-eyebrow {display:inline-flex;align-items:center;gap:9px;font:600 10px var(--mono);letter-spacing:.17em;color:#a4d6b3;}
.hero-eyebrow::before {content:'';width:7px;height:7px;background:#9bf0be;border-radius:50%;box-shadow:0 0 13px #78d69c;}
.hero-title {font-family:'Space Grotesk',sans-serif;color:#f2fff3;font-size:clamp(34px,3.5vw,53px);line-height:1.12;letter-spacing:-.07em;font-weight:650;max-width:640px;margin:24px 0 20px;}
.hero-title span {background:linear-gradient(100deg,#b9f5c1,#93cfae 48%,#d4f0b5);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero-description {font-size:14px;line-height:1.85;color:#b5c9bb;max-width:560px;}
.hero-visual {position:relative;display:flex;align-items:center;justify-content:center;min-height:325px;overflow:hidden;}
.orbit {position:absolute;border:1px solid #a8e4ba20;border-radius:50%;}
.orbit.one {width:315px;height:315px;}.orbit.two {width:246px;height:246px;border-color:#a8e4ba38;}.orbit.three {width:170px;height:170px;border-color:#b9ffca24;}
.orbit.one::before,.orbit.two::before {content:'';position:absolute;left:14%;top:14%;width:8px;height:8px;border-radius:50%;background:#a1eabb;box-shadow:0 0 22px #7ce0a5;}
.orbit.two::before {left:auto;right:3%;top:62%;width:6px;height:6px;}
.orbit-center {position:relative;display:flex;align-items:center;justify-content:center;flex-direction:column;width:128px;height:128px;border-radius:34px;background:linear-gradient(145deg,#294a39,#111f19);border:1px solid #7bcf9f55;box-shadow:0 0 80px #67d29e30,inset 0 1px #caffcd22;transform:rotate(-8deg);}
.orbit-center .glyph {font-size:44px;transform:rotate(8deg);filter:drop-shadow(0 0 20px #b8ffd866);}
.orbit-center small {font:600 9px var(--mono);color:#b6dfbe;letter-spacing:.14em;transform:rotate(8deg);}
.orbit-card {position:absolute;background:#0f1b17df;border:1px solid #3e624b;border-radius:12px;color:#c6ebce;box-shadow:0 12px 35px #0005;padding:9px 12px;font:600 10px var(--mono);letter-spacing:.02em;white-space:nowrap;}
.orbit-card.a {top:22%;right:6%;}.orbit-card.b {bottom:23%;left:3%;}.orbit-card.c {bottom:13%;right:5%;}
.orbit-label {position:absolute;top:10%;left:13%;font:500 10px var(--mono);color:#7da990;letter-spacing:.15em;}
.hero-foot {grid-column:1 / -1;padding:12px 25px;background:#09130eb0;border-top:1px solid #2b4536;display:flex;align-items:center;gap:9px;color:#a4cdb2;font:500 10px var(--mono);letter-spacing:.08em;}
.hero-foot .pipe {color:#567962;margin:0 6px;}
/* Metrics & sections */
.metrics {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin:0 0 35px;}
.metric-card {background:linear-gradient(145deg,#111a17,#0c1210);border:1px solid #26362e;border-radius:16px;padding:19px 22px;min-height:108px;}
.metric-number {font-family:'Space Grotesk',sans-serif;font-weight:650;color:#ddf4de;font-size:30px;letter-spacing:-.055em;line-height:1;}
.metric-label {font:600 10px var(--mono);color:#8fa99a;letter-spacing:.10em;margin-top:13px;}
.section-head {padding:15px 0 18px;}
.section-kicker {color:#8ccd9d;font:600 10px var(--mono);letter-spacing:.16em;}
.section-title {font:650 clamp(26px,2.6vw,35px)/1.2 'Space Grotesk',sans-serif;color:#eef7ef;letter-spacing:-.055em;margin:10px 0;}
.section-description {color:#99aea2;line-height:1.75;font-size:13px;max-width:850px;}
.mini-kicker {color:#90aa9a;font:600 10px var(--mono);letter-spacing:.15em;margin:19px 0 13px;}
.prompt-visual {border:1px solid #2b4034;border-radius:14px;padding:19px 17px 14px;background:linear-gradient(155deg,#16241b,#0d1713);min-height:120px;margin-bottom:8px;}
.prompt-icon {color:#a3dfb7;font-size:20px;display:block;margin-bottom:11px;}
.prompt-title {font:650 16px 'Space Grotesk',sans-serif;color:#e9f4ed;letter-spacing:-.03em;}
.prompt-desc {font-size:12px;line-height:1.5;color:#91a99a;margin-top:7px;}
.session-bar {display:flex;align-items:center;justify-content:space-between;margin:28px 0 15px;padding:0 0 13px;border-bottom:1px solid #25372d;color:#bdddc5;font:600 10px var(--mono);letter-spacing:.12em;}
.session-bar .count {color:#94a89a;font-weight:500;}
.empty-session {border:1px dashed #34503c;background:#0f191417;border-radius:15px;padding:28px 24px;color:#8fa898;text-align:center;font-size:13px;}
/* Streamlit primitives */
.stButton > button, .stLinkButton > a {background:#172920;border:1px solid #436a50;color:#def5e0;border-radius:11px;padding:0.6rem 0.85rem;transition:all .2s ease;box-shadow:none;font-weight:600;}
.stButton > button:hover,.stLinkButton > a:hover {background:#20392a;border-color:#8ecda0;color:#f2fff2;transform:translateY(-1px);}
.stButton > button:focus:not(:active),.stLinkButton > a:focus:not(:active) {border-color:#9cdbad;color:#f2fff2;box-shadow:0 0 0 2px #8cbb9e44;}
[data-baseweb="tab-list"] {gap:12px;border-bottom:1px solid #273b30;}
button[data-baseweb="tab"] {color:#9eafa2;font-weight:600;min-height:48px;padding:10px 16px;background:none;}
button[data-baseweb="tab"][aria-selected="true"] {color:#d3f4d8;background:#15231a;border-radius:10px 10px 0 0;}
[data-baseweb="tab-highlight"] {background-color:#92dca8;}
[data-testid="stChatMessage"] {background:#101a16;border:1px solid #2c3f33;border-radius:16px;padding:19px 20px;margin-bottom:14px;}
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {background:#233c2b;}
[data-testid="stChatInput"] {border:1px solid #496e57;border-radius:15px;background:#111d16;box-shadow:0 0 30px #5bad7630;}
[data-testid="stChatInput"] textarea {color:#e5f7eb;}
[data-testid="stChatInput"] button {color:#a9dcb7;}
.response-label {font:600 10px var(--mono);color:#91bca0;letter-spacing:.12em;margin-bottom:9px;}
[data-testid="stExpander"] {border:1px solid #2b4234;border-radius:12px;background:#0e1713;}
[data-testid="stTextInput"] input {background:#121f18;color:#e9f4eb;border:1px solid #3e5a48;border-radius:11px;}
[data-testid="stAlert"] {border-radius:12px;}
/* Library cards and architecture */
.paper-card {background:linear-gradient(148deg,#14221a,#0d1712);border:1px solid #2b4534;border-radius:16px;padding:22px 22px 17px;min-height:239px;margin:7px 0 9px;position:relative;overflow:hidden;}
.paper-card::before {content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#82d59d,#82d59d00);}
.paper-kicker {font:600 10px var(--mono);color:#91d5a6;letter-spacing:.13em;line-height:1.5;}
.paper-title {color:#e6f5eb;font:650 18px/1.35 'Space Grotesk',sans-serif;letter-spacing:-.03em;margin:15px 0 10px;}
.paper-meta {color:#8cac99;font:500 11px var(--mono);margin-bottom:12px;}
.paper-desc {color:#9db7a3;line-height:1.68;font-size:13px;}
.phase-card {background:#101c16;border:1px solid #304739;border-radius:15px;padding:20px 22px;min-height:180px;margin:8px 0 17px;}
.phase-name {color:#a6e1b4;font:600 10px var(--mono);letter-spacing:.12em;}
.phase-card p {color:#b7cdbb;line-height:1.7;font-size:13px;margin:20px 0 0;}
.arch-card {background:#111d16;border:1px solid #2c4534;border-radius:15px;padding:21px 24px;color:#a9c3ae;line-height:1.8;font-size:13px;margin:13px 0;}
.arch-card strong {color:#c5ebcf;font:600 10px var(--mono);letter-spacing:.12em;display:block;margin-bottom:12px;}
.arch-card code {color:#b2edc3;}
.arch-flow {background:#0b1510;border:1px solid #284333;border-radius:12px;color:#afd6b6;font:500 12px/1.9 var(--mono);padding:17px 21px;white-space:pre-wrap;overflow-wrap:anywhere;}
.moss-footer {border-top:1px solid #293b30;padding:20px 0;margin-top:34px;color:#779383;font:500 10px var(--mono);display:flex;flex-wrap:wrap;justify-content:space-between;gap:12px;letter-spacing:.04em;}
@media(max-width:990px) {.hero-shell{grid-template-columns:1fr;}.hero-copy{padding:36px 29px 10px;}.hero-visual{min-height:250px;}.orbit.one{width:235px;height:235px;}.orbit.two{width:190px;height:190px;}.orbit.three{width:132px;height:132px;}.orbit-center{width:102px;height:102px;border-radius:27px;}.orbit-card.a{right:12%;}.orbit-card.b{left:12%;}}
@media(max-width:600px) {.block-container{padding-top:1.3rem;}.hero-copy{padding:28px 22px 10px;}.hero-title{font-size:35px;}.metrics{grid-template-columns:1fr;gap:8px;}.metric-card{min-height:75px;padding:14px 18px;}.metric-number{font-size:25px;}.metric-label{margin-top:7px;}.hero-foot{flex-wrap:wrap;}.orbit-card{font-size:9px;}.orbit-card.a{right:3%;}.orbit-card.b{left:3%;}.section-title{font-size:27px;}.paper-card{min-height:0;}}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "pt"

papers = get_library()
provider_label = provider.upper()
model_name = os.getenv("LLM_MODEL", "")
has_key = usable_key(selected_key)
openalex_ready = usable_key(os.getenv("OPENALEX_API_KEY"))
langsmith_ready = bool(usable_key(os.getenv("LANGSMITH_API_KEY")) and os.getenv("LANGSMITH_TRACING", "").lower() == "true")

# O seletor fica sempre no conteúdo principal, inclusive com a sidebar fechada.
# Não depender da posição da sidebar, que o Streamlit pode recolher no celular.
header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown('<div class="moss-main-brand">✳ &nbsp; MOSS SCHOLAR <span>/ RESEARCH INTELLIGENCE</span></div>', unsafe_allow_html=True)
with header_right:
    st.selectbox(
        "🌐 Idioma / Language",
        options=list(LANGUAGES),
        format_func=lambda value: LANGUAGES[value],
        key="ui_lang",
    )
lang = st.session_state.ui_lang
t = I18N[lang]

with st.sidebar:
    st.markdown('''<div class="moss-brand"><div class="moss-mark">✳</div><div class="moss-wordmark">moss scholar<small>RESEARCH INTELLIGENCE</small></div></div>''', unsafe_allow_html=True)
    st.caption(f'🌐 {t["language"]}: {LANGUAGES[lang]}')
    st.markdown(f'''<div class="side-kicker">{html.escape(t["side_nav"])}</div>
    <div class="side-link"><b>✦</b>{html.escape(t["nav_assistant"])}</div>
    <div class="side-link"><b>▦</b>{html.escape(t["nav_library"])}</div>
    <div class="side-link"><b>◇</b>{html.escape(t["nav_about"])}</div>''', unsafe_allow_html=True)
    st.markdown(f'<div class="side-kicker">{html.escape(t["side_engine"])}</div>', unsafe_allow_html=True)
    model_html = html.escape((model_name or provider_label)[:58])
    st.markdown(f'''<div class="status-card">
      <div class="status-top">{html.escape(t["model"]).upper()}</div>
      <div class="status-model">{model_html}</div>
      <div class="status-row"><i class="status-dot {"" if has_key else "off"}"></i>{html.escape(t["configured"] if has_key else t["not_configured"])}</div>
      <div class="status-row"><i class="status-dot {"" if openalex_ready else "off"}"></i>{html.escape(t["openalex_yes"] if openalex_ready else t["openalex_no"])}</div>
      <div class="status-row"><i class="status-dot {"" if langsmith_ready else "off"}"></i>{html.escape(t["langsmith_yes"] if langsmith_ready else t["langsmith_no"])}</div>
    </div>''', unsafe_allow_html=True)
    # Mostrar só informações de configuração não sensíveis.
    st.caption(f"Provider: {provider_label} · Modelo: {model_name}")
    st.caption(f"Chave: {api_source} · Endpoint: {'openrouter.ai' if provider == 'openrouter' else 'api.openai.com'}")
    if provider_adjusted:
        st.warning({"pt": "Uma chave OpenRouter foi detectada; o provedor foi ajustado para OpenRouter.", "en": "An OpenRouter key was detected; the provider was set to OpenRouter.", "es": "Se detectó una clave OpenRouter; el proveedor se cambió a OpenRouter."}[lang])
    if secrets_status == "unreadable":
        st.error({"pt": "Não foi possível ler Streamlit Secrets; revise a sintaxe TOML.", "en": "Could not read Streamlit Secrets; check TOML syntax.", "es": "No se pudieron leer los Secrets; revisa la sintaxis TOML."}[lang])
    if st.button(f'↻  {t["reset"]}', use_container_width=True, key="reset_session"):
        st.session_state.history = []
        st.session_state.agent_messages = []
        st.session_state.pending_question = None
        st.rerun()
    st.markdown(f'<div class="side-footer">LANGGRAPH × LANGSMITH × STREAMLIT<br>{html.escape(t["side_foot"])}</div>', unsafe_allow_html=True)

# The layout is custom HTML/CSS, but inputs and interactions stay native to Streamlit.
main_status = t["hero_status_value"] if has_key else t["not_configured"]
st.markdown(f'''<div class="topline">
 <div class="topline-left"><strong>MOSS /</strong> {html.escape(t["workspace"])}</div>
 <div class="topline-right"><i></i> {html.escape(main_status).upper()}</div>
</div>
<div class="hero-shell">
 <div class="hero-copy">
  <div class="hero-eyebrow">{html.escape(t["eyebrow"])}</div>
  <div class="hero-title">{html.escape(t["hero_title"])}<br><span>{html.escape(t["hero_emphasis"])}</span></div>
  <div class="hero-description">{html.escape(t["hero_desc"])}</div>
 </div>
 <div class="hero-visual" aria-hidden="true">
  <div class="orbit-label">{html.escape(t["graph_tag"])}</div>
  <div class="orbit one"></div><div class="orbit two"></div><div class="orbit three"></div>
  <div class="orbit-center"><span class="glyph">✳</span><small>MOSS / AI</small></div>
  <div class="orbit-card a">◉ &nbsp; {html.escape(t["node1"])}</div>
  <div class="orbit-card b">⌁ &nbsp; {html.escape(t["node2"])}</div>
  <div class="orbit-card c">▦ &nbsp; {html.escape(t["node3"])}</div>
 </div>
 <div class="hero-foot"><span>● &nbsp; {html.escape(t["hero_status"])}</span><span class="pipe">/</span><span>REACT + LANGGRAPH</span><span class="pipe">/</span><span>{html.escape(t["graph_source"])}</span></div>
</div>
<div class="metrics">
 <div class="metric-card"><div class="metric-number">{len(papers):02d}</div><div class="metric-label">{html.escape(t["metric_papers"])}</div></div>
 <div class="metric-card"><div class="metric-number">03</div><div class="metric-label">{html.escape(t["metric_tools"])}</div></div>
 <div class="metric-card"><div class="metric-number">ReAct</div><div class="metric-label">{html.escape(t["metric_method"])}</div></div>
</div>''', unsafe_allow_html=True)

research_tab, library_tab, about_tab = st.tabs([t["tab_assistant"], t["tab_library"], t["tab_about"]])


def section_heading(kicker: str, title: str, description: str) -> None:
    st.markdown(f'''<div class="section-head"><div class="section-kicker">{html.escape(kicker)}</div>
       <div class="section-title">{html.escape(title)}</div>
       <div class="section-description">{html.escape(description)}</div></div>''', unsafe_allow_html=True)


def show_trace(events: list[dict], labels: dict) -> None:
    if not events:
        return
    with st.expander(f'{labels["trace"]} ({len(events)} {labels["trace_step"]})', expanded=False):
        for i, event in enumerate(events, start=1):
            st.markdown(f'**{i}. `{event["name"]}`**')
            with st.container(border=True):
                result = event["result"]
                st.code(result[:1100] + ("…" if len(result) > 1100 else ""), language="json")


def content_as_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(item.get("text", "") for item in value if isinstance(item, dict) and item.get("type") == "text")
    return str(value)


with research_tab:
    section_heading(t["section_eyebrow"], t["section_title"], t["section_desc"])
    st.markdown(f'<div class="mini-kicker">{html.escape(t["quick_eyebrow"])}</div>', unsafe_allow_html=True)
    prompt_icons = ("⌕", "⇄", "◈")
    cols = st.columns(3, gap="small")
    for i in range(3):
        with cols[i]:
            st.markdown(f'''<div class="prompt-visual">
                <span class="prompt-icon">{prompt_icons[i]}</span>
                <div class="prompt-title">{html.escape(t[f"prompt{i + 1}_title"])}</div>
                <div class="prompt-desc">{html.escape(t[f"prompt{i + 1}_desc"])}</div>
            </div>''', unsafe_allow_html=True)
            if st.button(t[f"prompt{i + 1}_title"], key=f"quick_prompt_{i}", use_container_width=True):
                st.session_state.pending_question = t[f"prompt{i + 1}"]
                st.rerun()

    session_len = len(st.session_state.history)
    st.markdown(f'<div class="session-bar"><span>⌁ &nbsp; {html.escape(t["session_label"])}</span><span class="count">{session_len:02d}</span></div>', unsafe_allow_html=True)

    # Keep the original ReAct agent and tool execution unchanged.
    # Process a pending question before displaying the history, so answers appear
    # above the persistent chat input after each rerun.
    question = st.session_state.pending_question
    if question:
        st.session_state.pending_question = None
        st.session_state.history.append({"role": "user", "content": question})
        if not has_key:
            reply = t["no_key"]
            trace = []
            status = "no_key"
        else:
            with st.spinner(t["working"]):
                try:
                    previous = st.session_state.agent_messages
                    graph = make_configured_graph()
                    response = graph.invoke(
                        {"messages": previous + [HumanMessage(content=question)]},
                        config={"recursion_limit": 24, "run_name": "MossScholar-ReAct"},
                    )
                    current = response["messages"]
                    new_messages = current[len(previous) + 1:]
                    reply = next(
                        (content_as_text(msg.content) for msg in reversed(current)
                         if isinstance(msg, AIMessage) and not msg.tool_calls),
                        t["no_answer"],
                    )
                    trace = [
                        {"name": msg.name or "tool", "result": content_as_text(msg.content)}
                        for msg in new_messages if isinstance(msg, ToolMessage)
                    ]
                    st.session_state.agent_messages = current
                    status = "ok"
                except Exception as exc:
                    # Do not display raw API exceptions; they may contain secrets.
                    category = type(exc).__name__
                    message = str(exc).lower()
                    if "401" in message or "invalid api key" in message or "authenticationerror" in category.lower():
                        if provider == "openrouter":
                            tip = {
                                "pt": "OpenRouter rejeitou a autenticação. Confira se a chave pertence ao OpenRouter, se está ativa e se o modelo e endpoint estão corretos. Provedor e origem da chave aparecem no painel lateral.",
                                "en": "OpenRouter authentication failed. Check that the key is active and belongs to OpenRouter, and verify the model and endpoint. Provider and key source appear in the sidebar.",
                                "es": "Falló la autenticación de OpenRouter. Comprueba que la clave esté activa y sea de OpenRouter y verifica el modelo y endpoint. El proveedor y origen de la clave aparecen en el panel lateral.",
                            }[lang]
                        else:
                            tip = t["error_401"]
                    elif "429" in message or "rate limit" in message:
                        tip = t["error_429"]
                    elif "tool" in message and ("support" in message or "unsupported" in message):
                        tip = t["error_tool"]
                    elif "recursion" in message:
                        tip = t["error_recursion"]
                    else:
                        tip = t["error_generic"]
                    reply = f'⚠️ {t["failure"]} ({category}). {tip}'
                    trace = []
                    status = "error"

        st.session_state.history.append({"role": "assistant", "content": reply, "trace": trace, "status": status})

    if not st.session_state.history:
        st.markdown(f'<div class="empty-session">✳ &nbsp; {html.escape(t["session_empty"])}</div>', unsafe_allow_html=True)

    for item in st.session_state.history:
        with st.chat_message(item["role"], avatar="🌿" if item["role"] == "assistant" else "👤"):
            label = t["assistant_label"] if item["role"] == "assistant" else t["you_label"]
            st.markdown(f'<div class="response-label">{html.escape(label)}</div>', unsafe_allow_html=True)
            st.markdown(item["content"])
            if item["role"] == "assistant":
                trace = item.get("trace", [])
                show_trace(trace, t)
                if not trace and item.get("status", "ok") == "ok":
                    st.caption(t["no_tools"])

with library_tab:
    section_heading(t["library_kicker"], t["library_title"], t["library_desc"])
    query = st.text_input(t["library_filter"], placeholder=t["library_placeholder"], key="paper_search")
    if query.strip():
        from papers import search_library
        shown = search_library(query, 10)
    else:
        shown = papers
    st.caption(f'{len(shown)} {t["library_found"]}')
    st.caption(t["summary_note"])
    for start in range(0, len(shown), 2):
        columns = st.columns(2)
        for index, paper in enumerate(shown[start:start + 2]):
            with columns[index]:
                title = html.escape(str(paper["title"]))
                topic = html.escape(str(paper["topic"]))
                # Use localized summaries when papers.py provides them. Preserve
                # the original Portuguese summary otherwise (do not invent one).
                summary_key = f"summary_{lang}"
                summary = html.escape(str(paper.get(summary_key) or paper.get("summary_pt") or ""))
                year = html.escape(str(paper["year"]))
                identifier = html.escape(str(paper["id"]))
                st.markdown(f'''<div class="paper-card">
                  <div class="paper-kicker">{topic}</div>
                  <div class="paper-title">{title}</div>
                  <div class="paper-meta">{year} &nbsp;·&nbsp; arXiv:{identifier}</div>
                  <div class="paper-desc">{summary}</div>
                </div>''', unsafe_allow_html=True)
                st.link_button(t["view_paper"], paper["url"], use_container_width=True)
                with st.expander(t["authors_methods"]):
                    st.write(", ".join(paper["authors"]))
                    st.caption(t["methods"] + ": " + ", ".join(paper["methods"]))

with about_tab:
    section_heading(t["about_kicker"], t["about_title"], t["about_desc"])
    phases = (("phase1", "phase1_desc"), ("phase2", "phase2_desc"), ("phase3", "phase3_desc"))
    for column, (phase_title, phase_desc) in zip(st.columns(3), phases):
        with column:
            st.markdown(f'''<div class="phase-card"><div class="phase-name">{html.escape(t[phase_title])}</div>
              <p>{html.escape(t[phase_desc])}</p></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="arch-card"><strong>{html.escape(t["tool_title"])}</strong>
      <code>search_papers</code> — {html.escape(t["tool_search"])}<br>
      <code>extract_information</code> — {html.escape(t["tool_extract"])}<br>
      <code>compare_studies</code> — {html.escape(t["tool_compare"])}
    </div>''', unsafe_allow_html=True)
    st.markdown('''<div class="arch-flow">START ──► agent (LLM) ──► tools_condition<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ├── tools: search / extract / compare ──┐<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; └── END (answer) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; │<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; agent ◄───────────────────────────────┘</div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="arch-card"><strong>{html.escape(t["observability"])}</strong>{html.escape(t["observability_desc"])}</div>
    <div class="arch-card"><strong>{html.escape(t["limitation_title"])}</strong>{html.escape(t["limitation_desc"])}</div>''', unsafe_allow_html=True)

st.markdown(f'''<div class="moss-footer"><span>© {html.escape(t["footer"])}</span><span>{html.escape(t["footer_right"])}</span></div>''', unsafe_allow_html=True)

# Native Streamlit input remains fixed at the bottom, as in the original app.
submitted = st.chat_input(t["chat_input"], key="moss_fixed_chat")
if submitted and submitted.strip():
    st.session_state.pending_question = submitted.strip()
    st.rerun()
