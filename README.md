# RAG — Ingestão e Busca em Documentos PDF

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17%20+%20pgvector-336791?logo=postgresql)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

Sistema de **RAG (Retrieval-Augmented Generation)** que permite fazer perguntas em linguagem natural sobre documentos PDF. O conteúdo dos arquivos é dividido em fragmentos (*chunks*), transformado em vetores numéricos (*embeddings*) via OpenAI e armazenado no PostgreSQL com a extensão **pgvector**. Na hora da consulta, os trechos mais relevantes são recuperados e enviados como contexto ao modelo GPT-4o-mini, que responde apenas com base nas informações presentes nos documentos.

## 📋 Tabela de Conteúdos

- [Features](#-features)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Usar](#-como-usar)
- [Configuração](#️-configuração)
- [Testes](#-testes)
- [Troubleshooting](#-troubleshooting)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)

---

## 🎯 Features

- **Ingestão de PDFs** — lê um ou mais arquivos `.pdf`, divide em *chunks* de 1 000 caracteres (sobreposição de 150) e indexa no banco vetorial.
- **Busca semântica** — recupera os 10 trechos mais relevantes usando similaridade de cosseno via `pgvector`.
- **Chat interativo com histórico** — sessão de conversa no terminal com memória de contexto gerenciada pelo LangChain.
- **Respostas fundamentadas** — o modelo responde **somente** com base nos documentos; nunca inventa informações externas.
- **Banco via Docker** — PostgreSQL 17 + extensão `vector` sobem com um único comando.
- **Limpeza automática** — a cada nova sessão de chat, os embeddings antigos são removidos e reprocessados, garantindo índice sempre atualizado.

---

## 📦 Pré-requisitos

| Ferramenta | Versão mínima | Observação |
|---|---|---|
| Python | 3.12+ | Testado em 3.14 |
| Docker + Docker Compose | Qualquer versão estável | Para subir o PostgreSQL |
| Conta OpenAI | — | Necessária para embeddings e chat |

> **Não tem Docker?** Instale em [docs.docker.com/get-docker](https://docs.docker.com/get-docker/).

---

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Abra o arquivo `.env` e preencha:

```dotenv
OPENAI_API_KEY=sk-...          # Obrigatório
```

> Os demais campos do `.env.example` são opcionais — o código usa valores padrão quando não fornecidos.

### 5. Suba o banco de dados

```bash
docker compose up -d
```

Aguarde os containers iniciarem (~10 segundos). O serviço `bootstrap_vector_ext` cria a extensão `vector` automaticamente.

---

## 📂 Estrutura do Projeto

```
mba-ia-desafio-ingestao-busca/
├── src/
│   ├── ingest.py        # Lê PDFs, cria chunks e salva embeddings no pgvector
│   ├── search.py        # Busca semântica no vector store
│   └── chat.py          # Chat interativo RAG no terminal
├── document.pdf         # Exemplo de documento para ingestão
├── docker-compose.yml   # PostgreSQL 17 + extensão pgvector
├── requirements.txt     # Dependências Python
├── .env.example         # Modelo de variáveis de ambiente
└── README.md
```

---

## 💻 Como Usar

### Passo 1 — Adicione seus PDFs

Coloque os arquivos `.pdf` que deseja indexar na **raiz do projeto** (junto ao `document.pdf` de exemplo). O módulo de ingestão detecta todos os arquivos `.pdf` nessa pasta automaticamente.

### Passo 2 — Inicie o chat

O chat executa a ingestão automaticamente ao ser iniciado:

```bash
python src/chat.py
```

Saída esperada:

```
[ChatRAG] ---> Iniciando chat RAG...
[ChatRAG] ---> Modelo de IA preparado.
[ChatRAG] ---> Limpando tabelas LangChain Postgres...
[ChatRAG] ---> Inserindo dados na tabela LangChain Postgres...
[ChatRAG] ---> Chat RAG no terminal — respostas com base nos documentos indexados.
[ChatRAG] ---> Comandos: sair | quit | exit | Ctrl+D para encerrar.

Você: _
```

### Passo 3 — Faça perguntas

```
Você: Qual é o valor de faturamento da empresa Alfa Telecom LTDA?
[ChatRAG] ---> Assistente: O faturamento da Alfa Telecom LTDA registrado no documento é...

Você: sair
[ChatRAG] ---> Até logo.
```

### Executar apenas a ingestão (sem chat)

```bash
python -c "from src.ingest import ingest_pdf; ingest_pdf()"
```

### Executar apenas uma busca pontual

```bash
python src/search.py
```

O script de exemplo busca: `"Qual é o valor de faturamento da empresa de nome Alfa Telecom LTDA ?"`. Edite a última linha do arquivo para testar outras perguntas.

---

## ⚙️ Configuração

### Variáveis de ambiente (`.env`)

| Variável | Padrão no código | Descrição |
|---|---|---|
| `OPENAI_API_KEY` | — | **Obrigatório.** Chave de API da OpenAI |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` | String de conexão ao PostgreSQL |
| `PG_VECTOR_COLLECTION_NAME` | `gpt5_collection` | Nome da coleção no pgvector |
| `PDF_PATH` | raiz do projeto | Diretório onde os PDFs são buscados |

### Parâmetros de chunking (`src/ingest.py`)

| Parâmetro | Valor padrão | Efeito |
|---|---|---|
| `chunk_size` | `1000` | Tamanho máximo de cada fragmento (caracteres) |
| `chunk_overlap` | `150` | Sobreposição entre fragmentos consecutivos |

### Modelo de linguagem (`src/chat.py`)

O modelo padrão é `gpt-4o-mini` com `temperature=1`. Para alterar, edite a classe `CallOpenAI`:

```python
return ChatOpenAI(model='gpt-4o', temperature=0.5)
```

---

## 🧪 Testes

O projeto não possui uma suíte de testes automatizados. Para validar o funcionamento end-to-end:

1. Certifique-se de que o Docker está rodando:

```bash
docker compose ps
```

2. Execute o chat e faça uma pergunta sobre o conteúdo do `document.pdf` incluído no repositório:

```bash
python src/chat.py
```

3. Se a resposta for coerente com o documento, o pipeline de ingestão + busca + geração está funcionando corretamente.

---

## 🐛 Troubleshooting

### `connection refused` ao iniciar o chat

O PostgreSQL ainda não está acessível. Aguarde os containers subirem e verifique:

```bash
docker compose ps
docker compose logs postgres
```

### `OPENAI_API_KEY` não encontrada

Certifique-se de que o arquivo `.env` existe na raiz do projeto e contém a chave:

```bash
cat .env
# OPENAI_API_KEY=sk-...
```

### Extensão `vector` não encontrada

O serviço `bootstrap_vector_ext` pode ter falhado. Execute manualmente:

```bash
docker exec -it postgres_rag psql -U postgres -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Resposta: "Não tenho informações necessárias para responder"

Significa que nenhum trecho relevante foi encontrado no índice. Verifique:
- Se os PDFs estão na raiz do projeto.
- Se a ingestão foi concluída sem erros (mensagem `Chunks inseridos no store.` deve aparecer no log).
- Se a pergunta diz respeito ao conteúdo dos documentos indexados.

### Nenhum PDF encontrado na ingestão

A classe `FilesDirectory` busca arquivos `.pdf` na **raiz do projeto** por padrão. Confirme que os arquivos têm extensão `.pdf` (minúsculo) e estão na pasta correta.

---

## 🤝 Contribuindo

1. Faça um *fork* do repositório.
2. Crie uma branch para sua feature: `git checkout -b feat/minha-feature`.
3. Faça o commit das alterações: `git commit -m "feat: descrição da mudança"`.
4. Envie para o seu fork: `git push origin feat/minha-feature`.
5. Abra um *Pull Request* descrevendo as mudanças.

---

## 📄 Licença

Distribuído para fins educacionais como parte do MBA em Inteligência Artificial. Consulte o responsável pelo repositório para informações sobre uso e redistribuição.
Descreva abaixo como executar a sua solução.
