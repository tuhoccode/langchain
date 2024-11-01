import pandas as pd
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from langchain.docstore.document import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
class lang:
    def __init__(self, embedding_file, file_csv, template, repo, key):
        self.file_csv = file_csv
        self.template = template
        self.repo = repo
        self.key  = key
        self.embedding_file = embedding_file
        self.data = pd.read_csv(self.file_csv)
        self.summary_title = dict(zip(self.data['summary'], self.data['title']))
        self.db = self.load_data()
        self.prompt = self.prompt()
        self.llm = self.llm()
        self.rag = self.RAG(self,self.load_data, self.prompt, self.llm)

    def load_data(self):
        document = [Document(page_content = row['summary']) for _, row in self.data.iterrows()]
        embeddings = HuggingFaceEmbeddings(model_name = self.embedding_file)
        db = FAISS.from_documents(document, embeddings)
        return db
    def prompt(self):
        template = PromptTemplate(template=self.template, input_variables = ['context','question'])
        return template
    def llm(self):
        llm = HuggingFaceHub(
            huggingfacehub_api_token = self.key,
            repo_id = self.repo,
            model_kwargs = {'temperature' : 1, 'max_length': 5000 }

        )

        return llm
    def RAG(self,db, template, llm):
        rag = RetrievalQA.from_chain_type(
            llm = llm,
            chain_typr = 'stuff',
            retriever = db.as_retriever(),
            chain_type_kwargs = {'prompt' : template}

        )   

        return rag
    
    def find_title(self, summary):
        text = [summary] + self.data['summary'].tolist()
        vectortf = TfidfVectorizer().fit_transform(text)
        vecto = vectortf.toarray()

        consine = cosine_similarity(vecto)
        consined = consine[0][1:]

        consined_arg = np.argmax(consined, axis = 1)
        consined_arged = consined[consined_arg]

        similar = 0.2

        if consined_arged > similar:
            summaryed = self.data['summary'].iloc[consined_arg]
            return self.summary_title.get(summaryed, 'not find title')
        return 'not find title'





