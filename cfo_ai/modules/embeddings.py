import os
import pickle
import tempfile

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from pypdf import PdfReader


class Embeddings:
    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        """
        Creates a directory to store the embeddings vectors
        """
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def storeDocEmbeds(self, file, filename, file_type):
        """
        Stores document embeddings using Langchain and FAISS
        """
        # Write the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name

        # Load the data from the file using Langchain
        if file_type == "csv":
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load_and_split()
            embeddings = OpenAIEmbeddings()
            vectors = FAISS.from_documents(data, embeddings)
        elif file_type == "pdf":
            reader = PdfReader(tmp_file_path)
            corpus = "".join(
                [p.extract_text() for p in reader.pages if p.extract_text()]
            )
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.split_text(corpus)
            embeddings = OpenAIEmbeddings()
            vectors = FAISS.from_texts(chunks, embeddings)

        os.remove(tmp_file_path)

        # Save the vectors to a pickle file
        with open(f"{self.PATH}/{filename}.pkl", "wb") as f:
            pickle.dump(vectors, f)

    def getDocEmbeds(self, file, filename, file_type):
        """
        Retrieves document embeddings
        """
        # Retrieve if already present
        if not os.path.isfile(f"{self.PATH}/{filename}.pkl"):
            # Otherwise store vectors
            self.storeDocEmbeds(file, filename, file_type)

        # Load vectors
        with open(f"{self.PATH}/{filename}.pkl", "rb") as f:
            vectors = pickle.load(f)
        return vectors
