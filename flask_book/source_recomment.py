# from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from langchain.docstore.document import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import imagehash
from PIL import Image

import numpy as np
class RecommentBook:
    def __init__(self, embedding_file, key, repo, csv_file, template):
        self.embedding = embedding_file
        self.key = key
        self.repo = repo
        self.csv_file = csv_file
        self.template = template
        self.data = pd.read_csv(self.csv_file)
        self.summary_title = dict(zip(self.data['summary'], self.data['title']))
        self.db = self.Documents()
        self.prompt = self.Prompt()
        self.llm = self.llms()
        self.rag = self.RAG(self.llm, self.prompt, self.db)

    def Documents(self):
        documents = [Document(page_content=row['summary']) for _, row in self.data.iterrows()]
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding)
        db = FAISS.from_documents(documents, embeddings)
        return db

    def Prompt(self):
        prompt = PromptTemplate(template=self.template, input_variables=['context', 'question'])
        return prompt

    def llms(self):
        llm = HuggingFaceHub(
            huggingfacehub_api_token=self.key,
            repo_id=self.repo,
            model_kwargs={'temperature': 1, 'max_length': 5000}
        )
        return llm

    def RAG(self, llm, prompt, db):
        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(),
            chain_type_kwargs={'prompt': prompt},
        )
        return rag

    # def find_title_summary(self, summary):
    #     query_document = Document(page_content=summary)
    #     result = self.db.similarity_search(query_document.page_content)
    #     not_query = result[0].page_content if result else None
    #     return self.summary_title.get(not_query, 'not found title')

    def find_title_summary(self, summary):
            documents = [summary] + self.data['summary'].tolist()

            vectorizer = TfidfVectorizer().fit_transform(documents)
            vectors = vectorizer.toarray()

            cosine_matrix = cosine_similarity(vectors)
            similar_indices = cosine_matrix[0][1:] 

            best_match_index = np.argmax(similar_indices)
            best_match_score = similar_indices[best_match_index]

            threshold = 0.2  
            if best_match_score > threshold:  
                matching_summary = self.data['summary'].iloc[best_match_index]
                return self.summary_title.get(matching_summary, 'not found title')

            return 'not found title'
    

    
    def compare_image(self, user_image_path):
        image_folder = "flask_book/static/images"  # Đường dẫn thư mục chứa ảnh
        user_image = Image.open(user_image_path)
        user_image_hash = imagehash.average_hash(user_image)  # Tạo hash cho ảnh người dùng

        # Duyệt qua các ảnh trong thư mục và so sánh
        for image_name in os.listdir(image_folder):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, image_name)
                image = Image.open(image_path)
                image_hash = imagehash.average_hash(image)  # Tạo hash cho ảnh trong thư mục

                # Tính toán độ tương đồng (khoảng cách Hamming)
                if user_image_hash - image_hash < 10:  # Giá trị ngưỡng có thể thay đổi
                    return f"Đã tìm thấy sách: {image_name.replace('.webp', '')}"

        return "Không tìm thấy sách phù hợp với hình ảnh."
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ImageRecognition:
    def __init__(self, image_folder="flask_book/static/images"):
        self.image_folder = image_folder
        # Tải model ResNet50 đã được pretrained
        self.model = models.resnet50(pretrained=True)
        # Xóa lớp fully connected cuối cùng để lấy feature vector
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Định nghĩa các bước transform ảnh
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Cache các feature vectors của ảnh trong thư mục
        self.image_features = {}
        self._precompute_features()

    def extract_features(self, image_path):
        """Trích xuất đặc trưng từ một ảnh"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            image = image.unsqueeze(0)  # Thêm batch dimension
            
            with torch.no_grad():
                features = self.model(image)
                features = features.squeeze().numpy()
                
            return features
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
            return None

    def _precompute_features(self):
        """Tính toán trước các feature vectors cho tất cả ảnh trong thư mục"""
        for image_name in os.listdir(self.image_folder):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_path = os.path.join(self.image_folder, image_name)
                features = self.extract_features(image_path)
                if features is not None:
                    self.image_features[image_name] = features

    def compare_images(self, query_image_path, threshold=0.8):
        """So sánh một ảnh query với tất cả ảnh trong thư mục"""
        # Trích xuất đặc trưng của ảnh query
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            return {"status": "error", "message": "Không thể xử lý ảnh đầu vào"}

        # So sánh với tất cả ảnh trong cache
        max_similarity = -1
        most_similar_image = None

        for image_name, features in self.image_features.items():
            similarity = cosine_similarity(
                query_features.reshape(1, -1),
                features.reshape(1, -1)
            )[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_image = image_name

        if max_similarity > threshold:
            return {
                "status": "found",
                "message": f"Đã tìm thấy ảnh tương tự: {most_similar_image}",
                "similarity": max_similarity,
                "matched_image": most_similar_image
            }
        else:
            return {
                "status": "not_found",
                "message": "Không tìm thấy ảnh tương tự",
                "similarity": max_similarity
            }

    def update_image_database(self):
        """Cập nhật lại cache feature vectors khi có ảnh mới được thêm vào thư mục"""
        self.image_features.clear()
        self._precompute_features()