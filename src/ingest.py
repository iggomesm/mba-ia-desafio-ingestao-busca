import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from pathlib import Path
import logging


class FilesDirectory:
    """Lista caminhos dos arquivos (não recursivo) em uma pasta `files`."""

    def __init__(self, files_dir: Path | str | None = None) -> None:
        if files_dir is None:
            self._root = Path(__file__).resolve().parent.parent
        else:
            self._root = Path(files_dir).resolve()

    def files(self) -> list[str]:
        if not self._root.is_dir():
            return []
        return sorted(
            str(p.resolve())
            for p in self._root.iterdir()
            if p.is_file() and p.suffix == ".pdf"
        )

class ClassLogger:
    def __init__(self, class_name):
        self.logger = logging.getLogger(class_name)
        logging.basicConfig(level=logging.INFO)
    
    def info(self, message):
        self.logger.info(f"\n[{self.logger.name}] ---> {message}\n")

class ImportDataPgVector():
    def __init__(self):
        load_dotenv()
        self.log = ClassLogger(self.__class__.__name__)
        
        self.log.info("Preparando parametrizacao de arquivos...")
        
        self.arquivos = FilesDirectory().files()
        self.ULR = 'postgresql+psycopg://postgres:postgres@localhost:5432/rag'
        self.PGVECTOR_COLLECTION='gpt5_collection'

    
    def create_chuncks_arquivo(self):
        chunks = []
        for arquivo in self.arquivos:
            loader = PyPDFLoader(file_path=arquivo)
            pagina = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
            chunk_overlap=150)

            chunks += splitter.split_documents(pagina)

        chunks_enriquecidos = [
            Document(
                page_content=d.page_content,
                metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
            )
            for d in chunks
        ]

        for chunck in chunks:
            self.log.info("\n" * 2)
            self.log.info("--------------------------------")
            self.log.info(chunck)
        self.log.info(len(chunks))

        return chunks_enriquecidos

    def save_data(self):
        self.log.info("Criando chunks do arquivo...")
        chuncks = self.create_chuncks_arquivo()
        
        self.log.info("Criando ids dos chunks...")
        ids = [f"doc-{i}" for i in range(len(chuncks))]
        
        self.log.info("Criando store...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        store = PGVector(
            embeddings=embeddings,
            collection_name=self.PGVECTOR_COLLECTION,
            connection=self.ULR,
            use_jsonb=True,
        )
        
        self.log.info("Inserindo chunks no store...")
        store.add_documents(documents=chuncks, ids=ids)  
        self.log.info("Chunks inseridos no store.")



def ingest_pdf():
    save_data = ImportDataPgVector()
    save_data.save_data()
    

if __name__ == "__main__":
    ingest_pdf()