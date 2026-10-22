# Moss Scholar 🌿

Um aplicativo Streamlit de pesquisa acadêmica com **agente ReAct**, **LangGraph** e **LangSmith**, usando a paleta **Black `#000000`** + **Moss Green `#8A9A5B`**.

## O que ele faz

- Mostra uma coleção inicial de **10 artigos reais do arXiv** sobre detecção de notícias falsas, machine learning e NLP.
- Aceita perguntas em português e pode pesquisar, extrair informações e comparar estudos.
- Apresenta as ferramentas efetivamente executadas depois de cada resposta.
- Oferece busca **adicional pelo OpenAlex** quando há uma chave configurada.
- Envia rastreamentos ao **LangSmith** quando habilitado (chave e variável `LANGSMITH_TRACING=true`).

## Arquitetura

```text
Usuário (Streamlit)
      ↓
    START
      ↓
 agent: LLM + bind_tools([search_papers, extract_information, compare_studies])
      ↓
 tools_condition
      ├─ fim → resposta
      └─ ferramentas → ToolNode → agent (ciclo ReAct)

LangSmith: tracing de chamadas do modelo, nós e ferramentas.
```

### Ferramentas obrigatórias

| Ferramenta | Ação |
|---|---|
| `search_papers(query, max_results)` | Busca artigos na coleção local; se configurado, também no OpenAlex. |
| `extract_information(reference)` | Obtém título, autores, ano, técnicas, link e tenta recuperar o resumo do arXiv. |
| `compare_studies(references)` | Monta comparação estruturada entre 2 e 5 artigos com dados disponíveis. |

> O agente escolhe quais chamar e pode repetir ferramentas. Não há roteiro rígido que force três chamadas em toda pergunta.

## Instalação local

Requer **Python 3.11+** e uma chave de um serviço de LLM que ofereça *tool calling* (OpenAI ou OpenRouter). Também requer chave do LangSmith para o requisito acadêmico de tracing.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Copie `.env.example` para `.env`, substitua as chaves de exemplo e execute:

```bash
streamlit run app.py
```

### Variáveis de ambiente

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=adicione_sua_chave
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=adicione_sua_chave_langsmith
LANGSMITH_PROJECT=moss-scholar-react
# OPENALEX_API_KEY=adicione_se_quiser_pesquisar_fora_da_biblioteca
```

**OpenRouter**: configure `LLM_PROVIDER=openrouter`, `LLM_MODEL=openai/gpt-4o-mini` (ou outro modelo com suporte a ferramentas) e `LLM_API_KEY` com a chave do OpenRouter. A integração usa um endpoint OpenAI-compatível.

**OpenAlex**: a coleção local não requer chave. A pesquisa externa exige `OPENALEX_API_KEY` e está sujeita aos limites da API. A chave gratuita pode ser obtida em https://openalex.org/settings/api.

## Como enviar ao GitHub

1. Crie um repositório novo no GitHub, por exemplo `moss-scholar-react`.
2. Extraia o ZIP deste projeto.
3. Faça upload de **todos os arquivos e pastas**, inclusive `.streamlit/config.toml`, `data/papers.json` e `.github/workflows/tests.yml`.
4. **Não faça commit** do arquivo `.env` nem de `.streamlit/secrets.toml`.
5. Para publicar, em https://share.streamlit.io conecte o repositório e indique `app.py` como arquivo principal.
6. Na área **Settings → Secrets**, insira as variáveis no formato de `.streamlit/secrets.toml.example` (sem valores fictícios).
7. No painel do LangSmith, abra o projeto `moss-scholar-react` e verifique os rastreamentos após fazer uma pergunta.

## Perguntas de demonstração

- `Encontre artigos introdutórios sobre detecção de fake news.`
- `Compare os estudos 2309.11052 e 2411.12703 sobre métodos de classificação.`
- `Quais artigos usam imagem e texto? Compare dois deles.`
- `O que o estudo LIAR descreve? Quais são os metadados?`

## Dez artigos iniciais

Os títulos e URLs estão em [`data/papers.json`](data/papers.json). Todos podem ser acessados gratuitamente no arXiv:

1. [LIAR: A New Benchmark Dataset](https://arxiv.org/abs/1705.00648)
2. [Fake News Detection on Social Media: A Data Mining Perspective](https://arxiv.org/abs/1708.01967)
3. [FakeNewsNet](https://arxiv.org/abs/1809.01286)
4. [Fakeddit](https://arxiv.org/abs/1911.03854)
5. [CSI: A Hybrid Deep Model](https://arxiv.org/abs/1703.06959)
6. [A Survey on Multimodal Disinformation Detection](https://arxiv.org/abs/2103.12541)
7. [Multimodal Fake News Detection](https://arxiv.org/abs/2112.04831)
8. [fakenewsbr: Brazilian Portuguese](https://arxiv.org/abs/2309.11052)
9. [Towards Reliable Misinformation Mitigation](https://arxiv.org/abs/2305.14928)
10. [Strengthening Fake News Detection: SVM vs BERT](https://arxiv.org/abs/2411.12703)

### Limitações importantes

- A biblioteca contém **metadados e sínteses editoriais**, não PDFs baixados nem indexação dos textos completos.
- A ferramenta de extração tenta recuperar o **abstract original do arXiv** por API; se falhar, informa que usou apenas a síntese inicial.
- Resultados externos do OpenAlex podem ter resumo ausente, e métodos não são preenchidos artificialmente.
- O agente não realiza checagem factual automática da qualidade dos resultados. Referências, números e conclusões precisam ser confirmados na fonte.
- O LangSmith só rastreia execuções quando a configuração estiver correta e o serviço estiver acessível.

## Testes

```bash
python -m unittest discover -s tests -v
```

Os testes verificam a integridade da biblioteca, busca local, extração e comparação; não utilizam chaves de IA.

## Referências técnicas

- https://docs.langchain.com/oss/python/langgraph/overview
- https://docs.langchain.com/oss/python/langchain/agents
- https://docs.langchain.com/langsmith/observability
- https://help.openalex.org/api/authentication/
