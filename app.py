# app.py - Complete Streamlit RAG Chatbot Application
import streamlit as st
import logging
import os
import json
import glob
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.schema import Document as LangChainDocument
from langchain_community.vectorstores import Qdrant, FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# Groq integration
from langchain_groq import ChatGroq

# PDF processing
from pypdf import PdfReader

# Vector store models (optional Qdrant support)
try:
    from qdrant_client.models import PointStruct
except ImportError:
    PointStruct = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

class Config:
    """Configuration management for the RAG chatbot application."""
    
    # API Keys
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "384"))
    
    # Qdrant Configuration
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
    QDRANT_MODE: str = os.getenv("QDRANT_MODE", ":memory:")  # or "localhost:6333" for persistent storage
    
    # Search Configuration
    SEARCH_LIMIT: int = int(os.getenv("SEARCH_LIMIT", "2"))
    
    # File Configuration
    PDF_PATH: str = os.getenv("PDF_PATH", "pdf/อาหารพื้นเมืองจังหวัดน่าน.pdf")
    
    # System Messages
    SYSTEM_MESSAGE: str = os.getenv("SYSTEM_MESSAGE", "คุณเป็นผู้ช่วยที่เชี่ยวชาญเกี่ยวกับข้อมูลในเอกสาร จงตอบคำถามอย่างกระชับและถูกต้อง")
    
    # LangChain Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TEXT_SPLITTER_METHOD: str = os.getenv("TEXT_SPLITTER_METHOD", "recursive")
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")
    QA_CHAIN_TYPE: str = os.getenv("QA_CHAIN_TYPE", "stuff")
    
    # Docling Configuration
    ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "true").lower() == "true"
    ENABLE_TABLE_EXTRACTION: bool = os.getenv("ENABLE_TABLE_EXTRACTION", "true").lower() == "true"
    
    # Advanced Features
    ENABLE_CONVERSATION_MEMORY: bool = os.getenv("ENABLE_CONVERSATION_MEMORY", "true").lower() == "true"
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required but not found in environment variables")
        return True

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Document:
    """Represents a document with its content and metadata."""
    id: int
    text: str
    metadata: Optional[dict] = None
    
    def to_point_struct(self, vector: List[float]):
        """Convert document to Qdrant PointStruct (if Qdrant is available)."""
        if PointStruct is None:
            return None
            
        payload = {"text": self.text}
        if self.metadata:
            payload.update(self.metadata)
        
        return PointStruct(
            id=self.id,
            vector=vector,
            payload=payload
        )

@dataclass
class SearchResult:
    """Represents a search result from Qdrant."""
    text: str
    score: float
    metadata: Optional[dict] = None

@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "system", "user", "assistant"
    content: str

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class PDFProcessor:
    """Handles PDF file processing operations."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            logger.info(f"Successfully extracted text from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    @staticmethod
    def prepare_documents_from_pdf(pdf_path: str) -> List[str]:
        """Prepare document chunks from PDF file."""
        try:
            text = PDFProcessor.extract_text_from_pdf(pdf_path)
            # Split by newlines and filter empty strings
            documents = [doc.strip() for doc in text.split("\n") if doc.strip()]
            
            logger.info(f"Prepared {len(documents)} document chunks from PDF")
            return documents
        except Exception as e:
            logger.error(f"Error preparing documents from PDF: {e}")
            raise

class TextChunker:
    """Handles text chunking strategies."""
    
    @staticmethod
    def chunk_by_sentences(text: str, chunk_size: int = 3) -> List[str]:
        """Chunk text by sentences."""
        sentences = text.split('.')
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk = '.'.join(sentences[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    @staticmethod
    def chunk_by_paragraphs(text: str) -> List[str]:
        """Chunk text by paragraphs."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

# =============================================================================
# DOCLING DOCUMENT LOADER
# =============================================================================

class DoclingDocumentLoader:
    """Advanced document loader using Docling for better PDF processing."""
    
    def __init__(self):
        self.converter = DocumentConverter()
        self._setup_converter()
    
    def _setup_converter(self):
        """Setup Docling converter with optimized settings."""
        try:
            # Configure PDF pipeline options
            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = True  # Enable OCR for scanned documents
            pdf_options.do_table_structure = True  # Extract table structures
            pdf_options.table_structure_options.do_cell_matching = True
            
            # Set pipeline options
            self.converter.upload_pipeline_options = {
                InputFormat.PDF: pdf_options
            }
            
            logger.info("Docling converter configured successfully")
        except Exception as e:
            logger.error(f"Error setting up Docling converter: {e}")
            raise
    
    def load_document(self, file_path: str) -> LangChainDocument:
        """Load and process document using Docling."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Convert document
            result = self.converter.convert(file_path)
            doc = result.document
            
            # Extract text content
            text_content = doc.export_to_markdown()
            
            # Create LangChain document
            langchain_doc = LangChainDocument(
                page_content=text_content,
                metadata={
                    "source": file_path,
                    "title": doc.name,
                    "file_type": Path(file_path).suffix,
                    "processing_method": "docling"
                }
            )
            
            logger.info(f"Successfully loaded document: {file_path}")
            return langchain_doc
            
        except Exception as e:
            logger.error(f"Error loading document with Docling: {e}")
            raise
    
    def load_multiple_documents(self, file_paths: List[str]) -> List[LangChainDocument]:
        """Load multiple documents."""
        documents = []
        for file_path in file_paths:
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} documents successfully")
        return documents

# =============================================================================
# TEXT SPLITTER
# =============================================================================

class AdvancedTextSplitter:
    """Advanced text splitting strategies."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize different splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_documents(self, documents: List[LangChainDocument], method: str = "recursive") -> List[LangChainDocument]:
        """Split documents using specified method."""
        try:
            if method == "recursive":
                split_docs = self.recursive_splitter.split_documents(documents)
            elif method == "token":
                split_docs = self.token_splitter.split_documents(documents)
            else:
                raise ValueError(f"Unknown splitting method: {method}")
            
            logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks using {method} method")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise

# =============================================================================
# LANGCHAIN RAG SERVICE
# =============================================================================

class LangChainRAGService:
    """Enhanced RAG service using LangChain framework."""
    
    def __init__(self):
        self.docling_loader = DoclingDocumentLoader()
        self.text_splitter = AdvancedTextSplitter()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            cache_folder="./model_cache"
        )
        self.vector_store: Optional[Any] = None
        self.llm = None
        self.qa_chain: Optional[Any] = None
        self.conversation_chain: Optional[Any] = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup Groq LLM."""
        try:
            Config.validate()
            self.llm = ChatGroq(
                groq_api_key=Config.GROQ_API_KEY,
                model_name=Config.LLM_MODEL,
                temperature=0.1
            )
            logger.info(f"Groq LLM initialized: {Config.LLM_MODEL}")
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            raise
    
    def create_vector_store(self, documents: List[LangChainDocument], store_type: str = "faiss"):
        """Create vector store from documents."""
        try:
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            
            # Create vector store based on type with fallback
            if store_type == "qdrant":
                try:
                    self.vector_store = Qdrant.from_documents(
                        documents=split_docs,
                        embedding=self.embeddings,
                        collection_name=Config.COLLECTION_NAME,
                        url=Config.QDRANT_MODE if Config.QDRANT_MODE != ":memory:" else None,
                        prefer_grpc=True
                    )
                    logger.info(f"Created Qdrant vector store with {len(split_docs)} documents")
                except Exception as qdrant_error:
                    logger.warning(f"Qdrant not available, falling back to FAISS: {qdrant_error}")
                    self.vector_store = FAISS.from_documents(
                        documents=split_docs,
                        embedding=self.embeddings
                    )
                    logger.info(f"Created FAISS vector store (fallback) with {len(split_docs)} documents")
            elif store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings
                )
                logger.info(f"Created FAISS vector store with {len(split_docs)} documents")
            elif store_type == "chroma":
                self.vector_store = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings,
                    persist_directory="./chroma_db"
                )
                logger.info(f"Created Chroma vector store with {len(split_docs)} documents")
            else:
                # Default to FAISS for unsupported types
                logger.warning(f"Unsupported vector store type: {store_type}, using FAISS")
                self.vector_store = FAISS.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings
                )
                logger.info(f"Created FAISS vector store (default) with {len(split_docs)} documents")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def setup_qa_chain(self, chain_type: str = "stuff"):
        """Setup QA chain for question answering."""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            # Custom prompt template
            prompt_template = """
            คุณเป็นผู้ช่วยที่เชี่ยวชาญเกี่ยวกับข้อมูลในเอกสาร จงตอบคำถามอย่างกระชับและถูกต้อง

            ข้อมูลอ้างอิง:
            {context}

            คำถาม: {question}
            คำตอบ:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=chain_type,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": Config.SEARCH_LIMIT}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            logger.info(f"QA chain setup completed with {chain_type} chain type")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
            raise
    
    def setup_conversation_chain(self):
        """Setup conversational retrieval chain."""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": Config.SEARCH_LIMIT}
                ),
                memory=self.memory,
                return_source_documents=True
            )
            
            logger.info("Conversational retrieval chain setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up conversation chain: {e}")
            raise
    
    def load_documents_from_file(self, file_path: str, store_type: str = "faiss"):
        """Load documents from file and create vector store."""
        try:
            # Load document using Docling
            document = self.docling_loader.load_document(file_path)
            
            # Create vector store
            self.create_vector_store([document], store_type)
            
            # Setup chains
            self.setup_qa_chain()
            self.setup_conversation_chain()
            
            logger.info(f"Successfully loaded document from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading documents from file: {e}")
            raise
    
    def answer_question(self, question: str, use_conversation: bool = False) -> Dict[str, Any]:
        """Answer question using RAG."""
        try:
            # Check if chains are initialized, if not try to initialize them
            if not self.qa_chain and not self.conversation_chain:
                logger.warning("QA chains not initialized, attempting to initialize...")
                
                # Try to load documents automatically if vector store exists
                if self.vector_store:
                    logger.info("Vector store exists, setting up chains...")
                    self.setup_qa_chain()
                    self.setup_conversation_chain()
                else:
                    # Try to load default PDF
                    pdf_files = glob.glob("pdf/*.pdf")
                    if pdf_files:
                        logger.info(f"Loading default PDF: {pdf_files[0]}")
                        self.load_documents_from_file(pdf_files[0])
                    else:
                        raise ValueError("QA chains not initialized and no PDF files found to load")
            
            if use_conversation and self.conversation_chain:
                # Use conversational chain
                result = self.conversation_chain.invoke({"question": question})
                answer = result["answer"]
                source_docs = result.get("source_documents", [])
            else:
                # Use simple QA chain
                if not self.qa_chain:
                    raise ValueError("QA chain not available")
                result = self.qa_chain.invoke({"query": question})
                answer = result["result"]
                source_docs = result.get("source_documents", [])
            
            # Extract source information
            sources = []
            for doc in source_docs:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
            
            response = {
                "answer": answer,
                "sources": sources,
                "question": question,
                "method": "conversation" if use_conversation else "qa"
            }
            
            logger.info("Successfully generated answer using LangChain RAG")
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        try:
            if not self.memory:
                return []
            
            history = []
            for message in self.memory.chat_memory.messages:
                history.append({
                    "role": message.__class__.__name__.lower().replace("message", ""),
                    "content": message.content
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        try:
            if self.memory:
                self.memory.clear()
            logger.info("Conversation history cleared")
            
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            raise

# =============================================================================
# UI FUNCTIONS
# =============================================================================

def get_langchain_rag_service():
    """Get cached LangChain RAG service instance."""
    try:
        # Validate configuration first
        Config.validate()
        return LangChainRAGService()
    except ValueError as e:
        st.error(f"❌ การตั้งค่าผิดพลาด: {e}")
        st.info("💡 กรุณาตั้งค่า GROQ_API_KEY ใน environment variables")
        return None
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการเริ่มต้นระบบ: {e}")
        logger.error(f"RAG service initialization error: {e}")
        return None

def get_pdf_files():
    """Get list of PDF files in ./pdf directory."""
    pdf_dir = "./pdf"
    if os.path.exists(pdf_dir):
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        return pdf_files
    return []

def show_rag_chatbot_page():
    """Display the main chatbot interface with LangChain RAG."""
    st.title("RAG Chatbot with LangChain")
    st.markdown("---")
    
    # Initialize LangChain RAG service
    rag_service = get_langchain_rag_service()
    
    if rag_service is None:
        st.error("❌ ไม่สามารถเริ่มต้นระบบได้ กรุณาตรวจสอบการตั้งค่า")
        return
    
    # Use default settings
    rag_mode = "Simple QA"
    vector_store_type = Config.VECTOR_STORE_TYPE
    chunk_size = Config.CHUNK_SIZE
    chunk_overlap = Config.CHUNK_OVERLAP
    search_limit = Config.SEARCH_LIMIT
    enable_ocr = Config.ENABLE_OCR
    enable_tables = Config.ENABLE_TABLE_EXTRACTION
    enable_memory = Config.ENABLE_CONVERSATION_MEMORY
    max_history = Config.MAX_CONVERSATION_HISTORY
    
    # Auto-load PDFs from ./pdf directory
    pdf_files = get_pdf_files()
    
    # Auto-load PDFs on startup if not already loaded
    if not st.session_state.get('documents_loaded', False) and pdf_files:
        with st.spinner("🔄 กำลังโหลดเอกสาร PDF อัตโนมัติ..."):
            try:
                success_count = 0
                for pdf_file in pdf_files:
                    try:
                        success = rag_service.load_documents_from_file(
                            pdf_file, 
                            store_type=vector_store_type
                        )
                        if success:
                            success_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load {pdf_file}: {e}")
                
                if success_count > 0:
                    st.session_state.documents_loaded = True
                    st.session_state.doc_count = success_count
                    st.success(f"✅ โหลดเอกสารอัตโนมัติสำเร็จ {success_count}/{len(pdf_files)} ไฟล์")
                else:
                    st.warning("⚠️ ไม่สามารถโหลดเอกสารใดได้")
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการโหลดอัตโนมัติ: {e}")
    
    # Check if documents are loaded
    if not st.session_state.get('documents_loaded', False):
        st.warning("📝 ไม่พบเอกสารในโฟลเดอร์ ./pdf กรุณาเพิ่มไฟล์ PDF")
        st.info("💡 เพิ่มไฟล์ PDF ในโฟลเดอร์ ./pdf แล้วรีเฟรชหน้า")
        
        # Show system status
        with st.expander("🔍 สถานะระบบ"):
            st.write("**RAG Service:**", "✅ พร้อมใช้งาน" if rag_service else "❌ ไม่พร้อมใช้งาน")
            st.write("**Vector Store:**", "✅ มีข้อมูล" if hasattr(rag_service, 'vector_store') and rag_service.vector_store else "❌ ไม่มีข้อมูล")
            st.write("**QA Chain:**", "✅ พร้อมใช้งาน" if hasattr(rag_service, 'qa_chain') and rag_service.qa_chain else "❌ ไม่พร้อมใช้งาน")
            st.write("**PDF Files:**", f"✅ พบ {len(pdf_files)} ไฟล์" if pdf_files else "❌ ไม่พบไฟล์")
            
            # Manual load button
            if pdf_files and rag_service:
                st.write("---")
                if st.button("🔄 โหลดเอกสารใหม่", use_container_width=True):
                    with st.spinner("🔄 กำลังโหลดเอกสาร..."):
                        try:
                            success_count = 0
                            for pdf_file in pdf_files:
                                try:
                                    success = rag_service.load_documents_from_file(
                                        pdf_file, 
                                        store_type=vector_store_type
                                    )
                                    if success:
                                        success_count += 1
                                except Exception as e:
                                    logger.warning(f"Failed to load {pdf_file}: {e}")
                            
                            if success_count > 0:
                                st.session_state.documents_loaded = True
                                st.session_state.doc_count = success_count
                                st.success(f"✅ โหลดเอกสารสำเร็จ {success_count}/{len(pdf_files)} ไฟล์")
                                st.rerun()
                            else:
                                st.error("❌ ไม่สามารถโหลดเอกสารได้")
                        except Exception as e:
                            st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        
        return
    
    # Show document status
    if st.session_state.get('documents_loaded', False):
        st.success(f"✅ เอกสารพร้อมใช้งาน ({st.session_state.get('doc_count', 0)} ไฟล์)")
        
        # Show detailed system status
        with st.expander("🔍 สถานะระบบ"):
            st.write("**RAG Service:**", "✅ พร้อมใช้งาน")
            st.write("**Vector Store:**", "✅ มีข้อมูล" if hasattr(rag_service, 'vector_store') and rag_service.vector_store else "❌ ไม่มีข้อมูล")
            st.write("**QA Chain:**", "✅ พร้อมใช้งาน" if hasattr(rag_service, 'qa_chain') and rag_service.qa_chain else "❌ ไม่พร้อมใช้งาน")
            st.write("**Conversation Chain:**", "✅ พร้อมใช้งาน" if hasattr(rag_service, 'conversation_chain') and rag_service.conversation_chain else "❌ ไม่พร้อมใช้งาน")
    
    # Chat Interface
    st.subheader("💬 แชทกับ Bot")
    
    # Initialize chat history
    if "enhanced_messages" not in st.session_state:
        st.session_state.enhanced_messages = []
    
    # Display chat history
    for message in st.session_state.enhanced_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message.get("sources"):
                with st.expander("📚 แหล่งข้อมูลอ้างอิง"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**แหล่งที่ {i}:**")
                        st.write(f"เนื้อหา: {source['content']}")
                        if source.get('metadata'):
                            st.write(f"ข้อมูลเพิ่มเติม: {source['metadata']}")
                        st.write("---")
    
    # Chat input
    if prompt := st.chat_input("พิมพ์คำถามของคุณ..."):
        # Add user message to chat history
        st.session_state.enhanced_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 กำลังคิดด้วย LangChain RAG..."):
                try:
                    use_conversation = rag_mode == "Conversational"
                    response = rag_service.answer_question(prompt, use_conversation=use_conversation)
                    
                    # Display answer
                    st.markdown(response["answer"])
                    
                    # Show sources
                    if response.get("sources"):
                        with st.expander("📚 แหล่งข้อมูลอ้างอิง"):
                            for i, source in enumerate(response["sources"], 1):
                                st.write(f"**แหล่งที่ {i}:**")
                                st.write(f"เนื้อหา: {source['content']}")
                                if source.get('metadata'):
                                    st.write(f"ข้อมูลเพิ่มเติม: {source['metadata']}")
                                st.write("---")
                    
                    # Add assistant response to chat history
                    st.session_state.enhanced_messages.append({
                        "role": "assistant", 
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                        "method": response.get("method", "unknown")
                    })
                    
                    # Update statistics
                    if "total_questions" not in st.session_state:
                        st.session_state.total_questions = 0
                    st.session_state.total_questions += 1
                    
                except ValueError as e:
                    error_msg = f"❌ ข้อผิดพลาดในการตั้งค่า: {e}"
                    st.error(error_msg)
                    st.info("💡 กรุณาตรวจสอบการตั้งค่า API keys และการโหลดเอกสาร")
                    logger.error(f"Configuration error: {e}")
                    st.session_state.enhanced_messages.append({"role": "assistant", "content": error_msg})
                    
                except Exception as e:
                    error_msg = f"❌ เกิดข้อผิดพลาดในการตอบคำถาม: {e}"
                    st.error(error_msg)
                    st.info("💡 กรุณาลองใหม่อีกครั้งหรือรีเฟรชระบบ")
                    logger.error(f"Question answering error: {e}")
                    import traceback
                    traceback.print_exc()
                    st.session_state.enhanced_messages.append({"role": "assistant", "content": error_msg})
    
    # Simple controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ ล้างประวัติ", use_container_width=True):
            st.session_state.enhanced_messages = []
            if hasattr(rag_service, 'clear_conversation_history'):
                rag_service.clear_conversation_history()
            st.rerun()
    
    with col2:
        if st.button("🔄 รีเฟรชระบบ", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Quick Questions with Enhanced Options
    st.subheader("❓ คำถามตัวอย่าง")
    
    # Categorized quick questions
    categories = {
        "🏛️ ประวัติศาสตร์": [
            "จังหวัดน่านมีประวัติศาสตร์อย่างไร?",
            "จังหวัดน่านเคยเป็นส่วนหนึ่งของอาณาจักรใดบ้าง?",
            "จังหวัดน่านมีเหตุการณ์สำคัญอะไรบ้างในอดีต?"
        ],
        "🏞️ สถานที่ท่องเที่ยว": [
            "จังหวัดน่านมีสถานที่ท่องเที่ยวอะไรบ้าง?",
            "จังหวัดน่านมีวัดสำคัญอะไรบ้าง?",
            "จังหวัดน่านมีธรรมชาติที่สวยงามอย่างไร?"
        ],
        "🍽️ อาหารและวัฒนธรรม": [
            "จังหวัดน่านมีอาหารพื้นเมืองอะไรบ้าง?",
            "จังหวัดน่านมีวัฒนธรรมอะไรที่น่าสนใจ?",
            "จังหวัดน่านมีงานประเพณีอะไรบ้าง?"
        ],
        "📊 ข้อมูลทั่วไป": [
            "จังหวัดน่านมีประชากรเท่าไหร่?",
            "จังหวัดน่านมีพื้นที่เท่าไหร่?",
            "จังหวัดน่านมีเขตการปกครองอะไรบ้าง?"
        ]
    }
    
    for category, questions in categories.items():
        with st.expander(category):
            cols = st.columns(2)
            for i, question in enumerate(questions):
                with cols[i % 2]:
                    if st.button(question, use_container_width=True, key=f"quick_{category}_{i}"):
                        # Add user message to chat history
                        st.session_state.enhanced_messages.append({"role": "user", "content": question})
                        
                        # Generate and add assistant response
                        with st.spinner("🤔 กำลังคิด..."):
                            try:
                                use_conversation = rag_mode == "Conversational"
                                response = rag_service.answer_question(question, use_conversation=use_conversation)
                                
                                # Add assistant response to chat history
                                st.session_state.enhanced_messages.append({
                                    "role": "assistant", 
                                    "content": response["answer"],
                                    "sources": response.get("sources", []),
                                    "method": response.get("method", "unknown")
                                })
                                
                                # Update statistics
                                if "total_questions" not in st.session_state:
                                    st.session_state.total_questions = 0
                                st.session_state.total_questions += 1
                                
                            except ValueError as e:
                                error_msg = f"❌ ข้อผิดพลาดในการตั้งค่า: {e}"
                                st.session_state.enhanced_messages.append({"role": "assistant", "content": error_msg})
                                logger.error(f"Quick question configuration error: {e}")
                                
                            except Exception as e:
                                error_msg = f"❌ เกิดข้อผิดพลาดในการตอบคำถาม: {e}"
                                st.session_state.enhanced_messages.append({"role": "assistant", "content": error_msg})
                                logger.error(f"Quick question error: {e}")
                        
                        st.rerun()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application - Chatbot only."""
    # Page configuration
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Hide the sidebar completely
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Show the chatbot page directly
    show_rag_chatbot_page()

if __name__ == "__main__":
    main()
