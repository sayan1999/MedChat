import sqlite3
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import faiss

from langchain_text_splitters import RecursiveCharacterTextSplitter
from joblib import dump, load
import os
from langchain_community.document_loaders import PyPDFLoader

import logging

logging.basicConfig(format="%(filename)s: %(message)s", level=logging.INFO)


class Embeddings:
    def __init__(self, config):
        self.sources = config["sources"]
        self.embeddings_on_disk = config["embeddings_on_disk"]
        if not os.path.isdir(self.embeddings_on_disk):
            os.mkdir(self.embeddings_on_disk)
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
        self.model_name = config["model"]
        self._get_or_create_embeddings()

    def _create_db_connection(self):
        conn = sqlite3.connect(os.path.join(self.embeddings_on_disk, "sqlite.db"))
        conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (idx INTEGER PRIMARY KEY, text TEXT)"""
        )
        return conn

    def _get_or_create_embeddings(self):
        if os.path.isdir(self.embeddings_on_disk) and os.path.isfile(
            os.path.join(self.embeddings_on_disk, "faiss.index")
        ):
            logging.info("Loading existing embeddings")
            self.model = SentenceTransformer(self.model_name)  # Load pre-trained model
            self.index = faiss.read_index(
                os.path.join(self.embeddings_on_disk, "faiss.index")
            )
            return

        splits = []
        logging.info("Splitting documents")
        for source in self.sources:
            loader = PyPDFLoader(source)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            doc_splits = text_splitter.split_documents(docs)
            splits.extend([doc.page_content for doc in doc_splits])

        conn = self._create_db_connection()
        for i, text in enumerate(splits):
            conn.execute(
                "INSERT INTO documents (idx, text) VALUES (?, ?)",
                (i, text),
            )
        conn.commit()

        logging.info("Processing embeddings")
        self.model = SentenceTransformer(self.model_name)  # Load pre-trained model
        embeddings = self.model.encode(splits)

        d = embeddings.shape[1]
        self.index = IndexFlatL2(d)
        self.index.add(embeddings)
        faiss.write_index(
            self.index, os.path.join(self.embeddings_on_disk, "faiss.index")
        )
        conn.close()
        return

    def search(self, query, n_top=10):
        logging.info("Searching db")
        query_embedding = self.model.encode([query])
        _, indexes = self.index.search(query_embedding, n_top)

        documents = []
        conn = self._create_db_connection()
        for idx in indexes[0]:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE idx = {}".format(idx))
            res = cursor.fetchone()
            if res:
                _, text = res
                documents.append(text)
        logging.debug(f"Found documents: {documents}")
        logging.info("Searching db Done")
        linebrks = "\n" + "*" * 80 + "\n"
        conn.close()
        return linebrks.join(documents)


# from google.colab import drive

# drive.mount("/content/drive")


CONFIG = {
    "embeddings": {
        "sources": [
            "[LANGE CURRENT Series] Stephen J. McPhee, Maxine Papadakis - CURRENT Medical Diagnosis and Treatment 2009 (LANGE CURRENT Series) (2008, McGraw-Hill Professional) - libgen.li.pdf",
            "(LANGE CURRENT Series) Norton Greenberger, Richard Blumberg, Robert Burakoff - CURRENT Diagnosis & Treatment Gastroenterology, Hepatology, & Endoscopy -McGraw-Hill Medical (2009).pdf",
        ],
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embeddings_on_disk": "/content/drive/My Drive/embeddings_db",
        "model": "all-MiniLM-L6-v2",
    }
}

db = Embeddings(CONFIG["embeddings"])
