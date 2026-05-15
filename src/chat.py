import os
import sys
from pathlib import Path

# Permite `python src/chat.py` a partir da raiz do repositório.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
import sys
import psycopg
from ingest import ImportDataPgVector
from search import FindDataPgVector
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import trim_messages

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:

"""

class ClassLogger:
    def __init__(self, class_name):
        self.logger = logging.getLogger(class_name)
        logging.basicConfig(level=logging.INFO)
    
    def info(self, message):
        self.logger.info(f"[{self.logger.name}] ---> {message}")

class ClearLangchainPgTables:
    def __init__(self):
        self.log = ClassLogger(self.__class__.__name__)
        self.conninfo = self._conninfo()

    def _conninfo(self) -> str:
        host = 'localhost'
        port = "5432"
        user = "postgres" 
        password = "postgres"
        dbname = "rag"
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    def delete_all(self) -> int:
        try:
            with psycopg.connect(self.conninfo) as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM langchain_pg_embedding")
                        n_emb = cur.rowcount
                        cur.execute("DELETE FROM langchain_pg_collection")
                        n_col = cur.rowcount
            self.log.info(f"Removidos {n_emb} registro(s) de langchain_pg_embedding.")
            self.log.info(f"Removidos {n_col} registro(s) de langchain_pg_collection.")
        except psycopg.Error as e:
            self.log.info(f"Erro ao acessar o PostgreSQL: {e}", file=sys.stderr)
            return 1
        return 0

class CallOpenAI():

    def __init__(self):
        self.__modelo = self.criar_conexao_modelo()

    @property
    def modelo(self):
        return self.__modelo

    def criar_conexao_modelo(self):
        load_dotenv()
        return ChatOpenAI(model='gpt-4o-mini', temperature=1)

class ChatRAG():
    def __init__(self):
        self.log = ClassLogger(self.__class__.__name__)
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            self.log.info("Defina OPENAI_API_KEY no arquivo .env para usar o chat.")
            sys.exit(1)
        
        self.log.info("Iniciando chat RAG...")
        self.llm = CallOpenAI().modelo
        self.log.info("Modelo de IA preparado.")
        self.clear_langchain_pg_tables()
        self.insert_data_langchain_pg()
        self.find_context = FindDataPgVector()

    def clear_langchain_pg_tables(self):
        self.log.info("Limpando tabelas LangChain Postgres...")
        ClearLangchainPgTables().delete_all()
        self.log.info("Tabelas LangChain Postgres limpas.")
    
    def insert_data_langchain_pg(self):
        self.log.info("Inserindo dados na tabela LangChain Postgres...")
        ImportDataPgVector().save_data()
        self.log.info("Dados inseridos na tabela LangChain Postgres.")

    def get_session_history(self, id_session):

        if id_session not in self.session_history:
            self.session_history[id_session] = InMemoryChatMessageHistory()

        return self.session_history[id_session]

    def retrieve_context(self, pergunta):
        self.log.info("Buscando contexto no vector store...")
        return self.find_context.find_prompt(pergunta)
    
    def prepare_inputs(self, payload: dict):
        raw_history = payload.get("linha_historico", [])
        contexto = payload.get("contexto", "")

        prompt_trim = trim_messages(
            raw_history,  
            token_counter=len,
            max_tokens=200,  
            strategy="last",  
            start_on="human", 
            include_system=True,
            allow_partial=False,
        )

        return {
            "pergunta": payload.get("pergunta", ""),
            "history": prompt_trim, 
            "contexto": contexto,
        }


    def main(self) -> None:
        
        self.log.info("Chat RAG no terminal — respostas com base nos documentos indexados.")
        self.log.info("Comandos: sair | quit | exit | Ctrl+D para encerrar.\n")
        self.session_history: dict[str, InMemoryChatMessageHistory] = {}
        config = {"configurable": {"session_id": "id_1"}}
        while True:
            try:
                pergunta = input("Você: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.log.info("\nEncerrando.")
                break

            if not pergunta:
                continue
            if pergunta.lower() in ("sair", "quit", "exit"):
                self.log.info("Até logo.")
                break

            try:
                contexto = self.retrieve_context(pergunta)
            except Exception as exc:
                self.log.info(f"Erro ao buscar contexto no vector store: {exc}\n")
                continue

            if not contexto.strip():
                self.log.info(
                    "Assistente: Não encontrei trechos no índice para essa pergunta. "
                    "Confira se a ingestão foi executada e o Postgres está acessível.\n"
                )
                continue

            prompt_preparado = RunnableLambda(self.prepare_inputs)
            
            chat_template = ChatPromptTemplate.from_messages(
                [  
                    (
                        "system",
                        PROMPT_TEMPLATE,
                    ),
                    MessagesPlaceholder("history"),
                    ("human", "{pergunta}"),
                ] 
            )  
            chain = prompt_preparado | chat_template | self.llm
            conversacao = RunnableWithMessageHistory(chain, self.get_session_history,
            input_messages_key="pergunta", history_messages_key="linha_historico")
            
            try:
                resposta = conversacao.invoke({'pergunta': pergunta, 'contexto': contexto},config=config)
                texto = resposta.content if hasattr(resposta, "content") else str(resposta)
                self.log.info(f"Assistente: {texto}\n")
            except Exception as exc:
                self.log.info(f"Erro ao chamar o modelo: {exc}\n")
                raise exc


if __name__ == "__main__":
    ChatRAG().main()