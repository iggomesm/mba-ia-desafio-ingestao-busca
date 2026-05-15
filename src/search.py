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
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
import logging

class ClassLogger:
    def __init__(self, class_name):
        self.logger = logging.getLogger(class_name)
        logging.basicConfig(level=logging.INFO)
    
    def info(self, message):
        self.logger.info(f"[{self.logger.name}] ---> {message}")

class FindDataPgVector():
    def __init__(self):
        load_dotenv()
        self.log = ClassLogger(self.__class__.__name__)
        
        self.log.info("Preparando parametrizacao de arquivos...")
        
        self.ULR = 'postgresql+psycopg://postgres:postgres@localhost:5432/rag'
        self.PGVECTOR_COLLECTION='gpt5_collection'

    def find_prompt(self, prompt):

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        store = PGVector(
            embeddings=embeddings,
            collection_name=self.PGVECTOR_COLLECTION,
            connection=self.ULR,
            use_jsonb=True,
        )

        self.log.info("Buscando dados...")
        resultado_busca = store.similarity_search_with_score(query=prompt, k=10)
        resultado_busca = sorted(resultado_busca, key=lambda item: item[1], reverse=True)
        contexto = "\n\n".join(
            f"{doc}\n" for doc in resultado_busca
        )

        return contexto

        # cont = 0
        # for doc, score in resultado_busca:
        #     self.log.info(f"\n")
        #     self.log.info(f"\n")
        #     self.log.info(f"===============================")
        #     self.log.info(f"[Score da busca] {score}")
        #     self.log.info(f"\n")
        #     linhas = doc.page_content.strip().splitlines()
        #     preview_doc = "\n".join(linhas[:7])
        #     self.log.info(f"[Documento] ---> {preview_doc}")
        #     self.log.info(f"\n")
        #     # Percorre cada par chave/valor nos metadados do documento (ex.: página, fonte).
        #     for k, v in doc.metadata.items():
        #         self.log.info(f"{k}: {v}")
        #     cont += 1
        #     if cont >= 10:
        #         break


def search_prompt(question=None):
    if question is None:
        return None

    find_data = FindDataPgVector()
    find_data.find_prompt(question)

if __name__ == "__main__":
    search_prompt("Qual é o valor de faturamento da empresa de nome Alfa Telecom LTDA ?")
        