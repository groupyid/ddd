import os
import fitz
import re
import faiss
import numpy as np
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import threading
import time
import pickle
from typing import List, Dict, Optional, Tuple
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-local storage
thread_local = threading.local()

# Konfigurasi API key Gemini
GEMINI_API_KEY = "AIzaSyBZdzvVraIbw4MwYNVxvPoF8i-gwTVs8C0"
genai.configure(api_key=GEMINI_API_KEY)

# Configuration
DOWNLOAD_FOLDER = 'D:/ProjectGemastik/AgroLLM/jurnal_ilmiah'
CACHE_FOLDER = 'D:/ProjectGemastik/AgroLLM/cache'
EMBEDDING_CACHE_FILE = os.path.join(CACHE_FOLDER, 'embeddings_cache.pkl')
INDEX_CACHE_FILE = os.path.join(CACHE_FOLDER, 'faiss_index.bin')
DOCS_CACHE_FILE = os.path.join(CACHE_FOLDER, 'documents_cache.pkl')

CHUNK_SIZE = 800  # Smaller chunks for better precision
CHUNK_OVERLAP = 150  # Higher overlap for context preservation
TOP_K_RETRIEVE = 5  # Retrieve fewer documents for faster response

# Create cache directory
os.makedirs(CACHE_FOLDER, exist_ok=True)

class OptimizedRAGSystem:
    def __init__(self):
        self.faiss_index = None
        self.doc_chunks = []
        self.model_gen = None
        self.embedding_cache = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
    def _load_cache(self):
        """Load cached embeddings and documents"""
        try:
            if os.path.exists(EMBEDDING_CACHE_FILE):
                with open(EMBEDDING_CACHE_FILE, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            
            if os.path.exists(DOCS_CACHE_FILE):
                with open(DOCS_CACHE_FILE, 'rb') as f:
                    self.doc_chunks = pickle.load(f)
                logger.info(f"Loaded {len(self.doc_chunks)} cached document chunks")
                
            if os.path.exists(INDEX_CACHE_FILE):
                self.faiss_index = faiss.read_index(INDEX_CACHE_FILE)
                logger.info("Loaded cached FAISS index")
                return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
        return False
    
    def _save_cache(self):
        """Save embeddings, documents, and index to cache"""
        try:
            with open(EMBEDDING_CACHE_FILE, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            
            with open(DOCS_CACHE_FILE, 'wb') as f:
                pickle.dump(self.doc_chunks, f)
            
            if self.faiss_index:
                faiss.write_index(self.faiss_index, INDEX_CACHE_FILE)
            
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _get_file_hash(self, filepath: str) -> str:
        """Generate hash for file to check if it's changed"""
        with open(filepath, 'rb') as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()

    def extract_text_from_pdf(self, file_path: str) -> List[Document]:
        """Optimized PDF text extraction with better title detection"""
        original_filename = os.path.basename(file_path)
        file_hash = self._get_file_hash(file_path)
        
        # Check if we already processed this file
        cache_key = f"{original_filename}_{file_hash}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key].get('documents', [])
        
        paper_title = self._extract_title_from_pdf(file_path)
        
        try:
            doc_pages = []
            with fitz.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages()):
                    page_text = page.get_text("text")
                    if page_text and len(page_text.strip()) > 50:  # Filter very short pages
                        # Clean and preprocess text
                        cleaned_text = self._clean_text(page_text)
                        if cleaned_text:
                            doc_pages.append(Document(
                                page_content=cleaned_text,
                                metadata={
                                    "title": paper_title, 
                                    "filename": original_filename, 
                                    "page": page_num + 1,
                                    "file_hash": file_hash
                                }
                            ))
            
            logger.info(f"Extracted {len(doc_pages)} pages from {paper_title}")
            return doc_pages
            
        except Exception as e:
            logger.error(f"Error reading file {original_filename}: {e}")
            return []

    def _extract_title_from_pdf(self, file_path: str) -> str:
        """Enhanced title extraction with better heuristics"""
        original_filename = os.path.basename(file_path)
        try:
            with fitz.open(file_path) as pdf:
                if len(pdf) == 0:
                    return original_filename
                
                first_page = pdf[0]
                blocks = first_page.get_text("dict")["blocks"]
                
                # Collect text with font information
                text_elements = []
                for block in blocks:
                    if 'lines' in block:
                        for line in block['lines']:
                            for span in line['spans']:
                                text = span['text'].strip()
                                if len(text) > 3 and not text.isdigit():
                                    text_elements.append({
                                        'text': text,
                                        'size': round(span['size']),
                                        'bbox': span['bbox']
                                    })
                
                # Find title candidates (largest font, top of page)
                if text_elements:
                    # Sort by font size and position
                    text_elements.sort(key=lambda x: (-x['size'], x['bbox'][1]))
                    
                    # Take the largest font text from the top third of the page
                    page_height = first_page.rect.height
                    top_third = page_height / 3
                    
                    title_candidates = [
                        elem for elem in text_elements[:5] 
                        if elem['bbox'][1] < top_third and len(elem['text'].split()) > 2
                    ]
                    
                    if title_candidates:
                        title = ' '.join([elem['text'] for elem in title_candidates[:2]])
                        title = re.sub(r'\s+', ' ', title).strip()
                        if len(title.split()) > 2:
                            return title
                
                return original_filename
        except Exception:
            return original_filename

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\b\d+\b(?=\s*$)', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\d\s]*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text if len(text) > 30 else ""

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[np.ndarray]:
        """Get embeddings in batches for better performance"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Add retry mechanism
                for attempt in range(3):
                    try:
                        result = genai.embed_content(
                            model="models/embedding-001", 
                            content=batch, 
                            task_type="retrieval_document"
                        )
                        batch_embeddings = [np.array(emb) for emb in result['embedding']]
                        embeddings.extend(batch_embeddings)
                        break
                    except Exception as e:
                        if attempt == 2:
                            logger.error(f"Failed to get embeddings after 3 attempts: {e}")
                            embeddings.extend([None] * len(batch))
                        else:
                            time.sleep(1)  # Wait before retry
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                embeddings.extend([None] * len(batch))
        
        return embeddings

    def get_query_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get query embedding with caching"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            result = genai.embed_content(
                model="models/embedding-001", 
                content=text, 
                task_type="retrieval_query"
            )
            embedding = np.array(result['embedding'])
            self.embedding_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            return None

    def semantic_search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> Tuple[List[str], Dict[str, str]]:
        """Enhanced semantic search with re-ranking"""
        query_embedding = self.get_query_embedding(query)
        
        if query_embedding is None or self.faiss_index is None:
            return [], {}
        
        # Search in FAISS index
        distances, indices = self.faiss_index.search(
            np.array([query_embedding]).astype('float32'), 
            top_k * 2  # Get more candidates for re-ranking
        )
        
        # Collect context chunks and sources
        context_chunks = []
        unique_sources = {}
        seen_content = set()
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.doc_chunks):
                chunk = self.doc_chunks[idx]
                content = chunk.page_content
                
                # Avoid duplicate content
                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    
                    # Add similarity score to metadata for potential re-ranking
                    similarity_score = 1 / (1 + distances[0][i])  # Convert distance to similarity
                    
                    context_chunks.append({
                        'content': content,
                        'score': similarity_score,
                        'metadata': chunk.metadata
                    })
                    
                    filename = chunk.metadata['filename']
                    title = chunk.metadata['title']
                    unique_sources[filename] = title  # key=filename, value=title (opsional)
        
        # Re-rank by relevance (simple keyword matching + semantic score)
        query_words = set(query.lower().split())
        for chunk in context_chunks:
            content_words = set(chunk['content'].lower().split())
            keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
            chunk['final_score'] = chunk['score'] * 0.7 + keyword_overlap * 0.3
        
        # Sort by final score and take top_k
        context_chunks.sort(key=lambda x: x['final_score'], reverse=True)
        context_chunks = context_chunks[:top_k]
        
        # Extract content
        final_contexts = [chunk['content'] for chunk in context_chunks]
        
        return final_contexts, unique_sources

    def generate_response(self, user_question: str, context: str, history: str) -> str:
        """Generate response with optimized prompt"""
        prompt = f"""Sebagai Penyuluh Pertanian Digital yang ahli, jawab pertanyaan petani berikut berdasarkan konteks penelitian yang diberikan.

ATURAN PENTING:
- Jawaban HARUS berdasarkan konteks yang diberikan
- Berikan jawaban yang praktis dan dapat diterapkan
- Jika informasi tidak lengkap, katakan "Informasi lebih detail tidak tersedia dalam dokumen"
- Gunakan bahasa yang mudah dipahami petani
- Fokus pada solusi praktis
- Kalo ada istilah yang sulit, jelaskan dengan sederhana

RIWAYAT PERCAKAPAN SEBELUMNYA:
{history}

KONTEKS PENELITIAN:
{context}

PERTANYAAN: {user_question}

JAWABAN PRAKTIS:"""

        try:
            response = self.model_gen.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more consistent responses
                    top_p=0.8,
                    top_k=40
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."

# Global instance
rag_system = OptimizedRAGSystem()

def initialize_rag_system():
    """Initialize the RAG system with caching and optimization"""
    global rag_system
    
    if not hasattr(thread_local, 'initialized'):
        logger.info("Initializing optimized RAG system...")
        
        # Try to load from cache first
        if rag_system._load_cache() and rag_system.faiss_index is not None:
            logger.info("✅ RAG system loaded from cache")
            rag_system.model_gen = genai.GenerativeModel("gemini-2.5-flash")
            thread_local.initialized = True
            return
        
        # If cache not available, build from scratch
        logger.info(f"Building RAG system from folder: {DOWNLOAD_FOLDER}...")
        
        pdf_files = [
            os.path.join(DOWNLOAD_FOLDER, f) 
            for f in os.listdir(DOWNLOAD_FOLDER) 
            if f.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            logger.error("❌ No PDF files found.")
            return
        
        # Process PDFs with parallel processing
        all_documents = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(rag_system.extract_text_from_pdf, pdf_file): pdf_file 
                for pdf_file in pdf_files
            }
            
            for future in future_to_file:
                docs = future.result()
                if docs:
                    all_documents.extend(docs)
                    logger.info(f"  -> Processed: {docs[0].metadata['title']}")
        
        if not all_documents:
            logger.error("❌ No documents processed successfully.")
            return
        
        # Split documents into chunks
        rag_system.doc_chunks = rag_system.text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(rag_system.doc_chunks)} text chunks")
        
        # Generate embeddings in batches
        chunk_texts = [chunk.page_content for chunk in rag_system.doc_chunks]
        embeddings_list = rag_system.get_embeddings_batch(chunk_texts)
        
        # Filter valid embeddings
        valid_embeddings = []
        valid_chunks = []
        for i, emb in enumerate(embeddings_list):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_chunks.append(rag_system.doc_chunks[i])
        
        if not valid_embeddings:
            logger.error("❌ No valid embeddings generated.")
            return
        
        rag_system.doc_chunks = valid_chunks
        
        # Create FAISS index
        dimension = valid_embeddings[0].shape[0]
        rag_system.faiss_index = faiss.IndexFlatIP(dimension)  # Use Inner Product for better performance
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(valid_embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        rag_system.faiss_index.add(embeddings_array)
        
        # Initialize generation model
        rag_system.model_gen = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Save to cache
        rag_system._save_cache()
        
        thread_local.initialized = True
        logger.info("✅ Optimized RAG system ready to use.")

def get_rag_response(user_question: str, history: List[Dict] = None) -> Tuple[str, List[Dict]]:
    """Get response from optimized RAG system"""
    if not hasattr(thread_local, 'initialized'):
        initialize_rag_system()
        if not hasattr(thread_local, 'initialized'):
            return "Sistem belum siap digunakan. Silakan coba lagi nanti.", []
    
    try:
        # Prepare chat history
        chat_history = ""
        if history:
            chat_history = "".join([
                f"Petani: {turn['user']}\nAsisten: {turn['bot']}\n\n" 
                for turn in history[-3:]  # Only use last 3 turns for context
            ])
        
        # Get relevant context
        context_chunks, unique_sources = rag_system.semantic_search(user_question)
        
        if not context_chunks:
            return "Maaf, tidak dapat menemukan informasi yang relevan dalam dokumen.", []
        
        context = "\n\n---\n\n".join(context_chunks)
        
        # Generate response
        response = rag_system.generate_response(user_question, context, chat_history)


        # Cek jika jawaban adalah template "tidak tersedia", "tidak dapat menemukan informasi", atau permintaan klarifikasi
        response_lower = response.lower()
        no_info_phrases = [
            "tidak dapat menemukan informasi yang relevan",
            "informasi lebih detail tidak tersedia dalam dokumen",
            "maaf, tidak dapat menemukan informasi yang relevan dalam dokumen",
            "maaf, terjadi kesalahan saat memproses pertanyaan anda. silakan coba lagi.",
            "pertanyaan yang bapak/ibu sampaikan",
            "tidak jelas dan tidak dapat kami pahami",
            "mohon sampaikan pertanyaan bapak/ibu dengan lebih jelas"
        ]
        if any(phrase in response_lower for phrase in no_info_phrases):
            return response, []

        # Format sources dengan url ke file PDF
        sources = [
            {
                "display_name": v if v else filename,  # tampilkan judul jika ada, fallback ke nama file
                "filename": filename,
                "url": f"/static/jurnal_ilmiah/{filename}",
            }
            for filename, v in unique_sources.items()
        ]
        return response, sources
        
    except Exception as e:
        logger.error(f"Error in get_rag_response: {e}")
        return "Terjadi error di server. Silakan coba lagi.", []