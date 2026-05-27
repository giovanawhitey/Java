import os
import shutil
from pathlib import Path

from fastapi import FastAPI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["ANONYMIZED_TELEMETRY"] = "False"

app = FastAPI()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://ollama:11434"))
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./db_local")
RECRIAR_DB = os.getenv("RAG_RECRIAR_DB", "true").lower() == "true"

qa_chain = None
documentos_carregados = []
total_chunks = 0


def listar_documentos():
    documentos_env = os.getenv("RAG_DOCUMENTOS", "").strip()

    if documentos_env:
        return [Path(nome.strip()) for nome in documentos_env.split(",") if nome.strip()]

    preferidos = [
        Path("DocumentacaoGrupo2_atualizada.pdf"),
        Path("documentacao.pdf"),
        Path("documento2.pdf"),
    ]

    encontrados = [arquivo for arquivo in preferidos if arquivo.exists()]
    if encontrados:
        return encontrados

    return sorted(Path(".").glob("*.pdf"))


def carregar_pdfs():
    documentos = []

    for caminho in listar_documentos():
        if not caminho.exists():
            print(f"AVISO: documento nao encontrado: {caminho}")
            continue

        print(f"Lendo documento: {caminho}")
        loader = PyPDFLoader(str(caminho))
        paginas = loader.load()

        for pagina in paginas:
            pagina.metadata["arquivo"] = caminho.name
            pagina.metadata["pagina"] = pagina.metadata.get("page", 0) + 1

        documentos.extend(paginas)
        documentos_carregados.append(caminho.name)

    return documentos


def montar_rag():
    global qa_chain, total_chunks

    print(">>> Iniciando RAG EasyData com Ollama <<<")

    documentos = carregar_pdfs()
    if not documentos:
        print("ERRO: nenhum documento PDF foi carregado.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=180,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documentos)
    total_chunks = len(chunks)

    if RECRIAR_DB and Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    print(f"Documentos carregados: {documentos_carregados}")
    print(f"Chunks criados: {total_chunks}")
    print("--- Criando banco vetorial ChromaDB ---")

    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    llm = ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="easydata_documentacao",
    )

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
        },
    )

    prompt = ChatPromptTemplate.from_template("""
Voce e o Assistente EasyData, chatbot do site institucional da EasyData.

Use somente as informacoes do contexto fornecido.
Responda em portugues do Brasil, de forma clara, direta e amigavel.
Se a resposta nao estiver no contexto, diga: "Nao encontrei essa informacao na documentacao do projeto."
Nao invente dados, integrantes, numeros, tecnologias, resultados ou promessas.
Quando a pergunta for ampla, resuma primeiro e depois detalhe em poucos topicos.

Contexto:
{context}

Pergunta:
{input}

Resposta:
""")

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("--- TUDO PRONTO: Sistema de busca ativo! ---")


try:
    montar_rag()
except Exception as e:
    print(f"ERRO CRITICO NA INICIALIZACAO: {str(e)}")


@app.get("/")
def home():
    return {
        "status": "Online" if qa_chain else "Sistema nao inicializado",
        "documentos": documentos_carregados,
        "chunks": total_chunks,
        "ollama": OLLAMA_BASE_URL,
        "embed_model": OLLAMA_EMBED_MODEL,
        "chat_model": OLLAMA_CHAT_MODEL,
    }


@app.get("/ask")
async def ask(question: str):
    if qa_chain is None:
        return {"erro": "O sistema nao foi inicializado."}

    if not question or not question.strip():
        return {"erro": "Envie uma pergunta valida."}

    try:
        print(f"Pergunta recebida: {question}")
        result = qa_chain.invoke({"input": question})

        fontes = []
        for doc in result.get("context", []):
            metadata = doc.metadata
            fonte = {
                "arquivo": metadata.get("arquivo", metadata.get("source", "")),
                "pagina": metadata.get("pagina", metadata.get("page", "")),
            }
            if fonte not in fontes:
                fontes.append(fonte)

        return {
            "resposta": result.get("answer", ""),
            "fontes": fontes[:3],
        }

    except Exception as e:
        print(f"Erro ao processar pergunta: {str(e)}")
        return {"erro": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)