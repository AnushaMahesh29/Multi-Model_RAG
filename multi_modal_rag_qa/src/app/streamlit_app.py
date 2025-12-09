"""
Streamlit App
Interactive web UI for the Multi-Modal RAG QA System.
"""

import streamlit as st
import sys
from pathlib import Path
import os
import tempfile

# Add parent directories to path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "rag"))
sys.path.insert(0, str(current_dir / "embeddings"))
sys.path.insert(0, str(current_dir / "vectorstore"))
sys.path.insert(0, str(current_dir / "preprocessing"))
sys.path.insert(0, str(current_dir / "ingestion"))

try:
    from rag.pipeline import RAGPipeline
    from ingestion.pdf_loader import PDFLoader
    from ingestion.text_extractor import TextExtractor
    from ingestion.table_extractor import TableExtractor
    from ingestion.image_extractor import ImageExtractor
    from ingestion.ocr_processor import OCRProcessor
    from preprocessing.cleaner import TextCleaner
    from preprocessing.chunker import TextChunker
    from embeddings.embed_pipeline import EmbeddingPipeline
    from vectorstore.faiss_builder import FAISSBuilder
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure all modules are in the correct location.")
    st.error(f"Current directory: {current_dir}")
    st.error(f"Python path: {sys.path}")
    st.stop()


def process_pdf(pdf_file, progress_bar, status_text):
    """Process uploaded PDF and build index."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            pdf_path = tmp_file.name
        
        status_text.text("Loading PDF...")
        progress_bar.progress(10)
        
        # Create data directories if they don't exist
        Path("data/intermediate/images").mkdir(parents=True, exist_ok=True)
        Path("data/index").mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load PDF
        loader = PDFLoader(pdf_path)
        loader.load()
        metadata = loader.get_metadata()
        
        status_text.text(f"Extracting text from {metadata['page_count']} pages...")
        progress_bar.progress(20)
            
        # Step 2: Extract text
        text_extractor = TextExtractor(loader.doc)
        all_text = text_extractor.extract_text_from_all_pages()
        
        status_text.text("Extracting tables...")
        progress_bar.progress(35)
        
        # Step 3: Extract tables (use stream flavor for better results with borderless tables)
        table_extractor = TableExtractor(pdf_path)
        all_tables = table_extractor.extract_tables_from_all_pages(flavor="stream")
        
        status_text.text("Extracting images...")
        progress_bar.progress(50)
        
        # Step 4: Extract images
        image_extractor = ImageExtractor(loader.doc)
        all_images = image_extractor.extract_images_from_all_pages(save_images=True)
        
        status_text.text("Running OCR on images...")
        progress_bar.progress(60)
        
        # Step 5: OCR
        ocr = OCRProcessor()
        ocr_results = ocr.process_images_from_directory("data/intermediate/images")
        
        # Add OCR text to images
        ocr_idx = 0
        for page_data in all_images:
            for img in page_data["images"]:
                if ocr_idx < len(ocr_results):
                    img["ocr_text"] = ocr_results[ocr_idx]["text"]
                    ocr_idx += 1
        
        status_text.text("Cleaning and chunking text...")
        progress_bar.progress(70)
        
        # Step 6: Clean and chunk
        cleaner = TextCleaner()
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        
        text_chunks = []
        for page_text in all_text:
            if page_text["text"]:
                cleaned = cleaner.clean(page_text["text"])
                chunks = chunker.chunk_text(
                    cleaned, metadata={"page_num": page_text["page_num"]}
                )
                text_chunks.extend(chunks)
        
        # Prepare image data
        image_data = []
        for page_data in all_images:
            for img in page_data["images"]:
                image_data.append({
                    "page_num": page_data["page_num"],
                    "image_num": img["image_num"],
                    "saved_path": img.get("saved_path"),
                    "ocr_text": img.get("ocr_text", ""),
                })
        
        status_text.text("Generating embeddings...")
        progress_bar.progress(80)
        
        # Step 7: Generate embeddings
        embed_pipeline = EmbeddingPipeline(device="cpu")
        embedded_data = embed_pipeline.embed_all(
            text_chunks=text_chunks,
            image_data=image_data,
            table_data=all_tables
        )
        
        status_text.text("Building FAISS index...")
        progress_bar.progress(90)
        
        # Step 8: Build FAISS index
        combined = embed_pipeline.combine_embeddings(embedded_data)
        embeddings_array = embed_pipeline.get_embeddings_array(combined)
        
        builder = FAISSBuilder(
            embedding_dim=embeddings_array.shape[1],
            index_type="flat"
        )
        builder.add_from_embedded_items(combined)
        
        # Step 9: Save index
        index_dir = Path("data/index")
        index_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = index_dir / "faiss_index.bin"
        metadata_path = index_dir / "metadata.pkl"
        
        builder.save(str(index_path), str(metadata_path))
        
        status_text.text("Processing complete!")
        progress_bar.progress(100)
        
        # Close PDF
        loader.close()
        
        # Clean up temp file
        try:
            os.unlink(pdf_path)
        except:
            pass
        
        return {
            "success": True,
            "total_vectors": builder.index.ntotal,
            "text_chunks": len(text_chunks),
            "images": len(image_data),
            "tables": len(all_tables),
            "pages": metadata['page_count']
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        status_text.text(f"Error: {str(e)}")
        st.error(f"Detailed error:\n{error_details}")
        return {"success": False, "error": str(e)}


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    index_path = "data/index/faiss_index.bin"
    metadata_path = "data/index/metadata.pkl"

    if not Path(index_path).exists() or not Path(metadata_path).exists():
        return None

    # Get API key from environment or session state
    api_key = os.getenv("GROQ_API_KEY") or st.session_state.get("groq_api_key")

    if not api_key:
        st.error("GROQ_API_KEY not found. Please set it in the sidebar.")
        return None

    try:
        pipeline = RAGPipeline(
            index_path=index_path,
            metadata_path=metadata_path,
            groq_api_key=api_key,
            llm_model=st.session_state.get("llm_model", "llama-3.1-70b-versatile"),
        )
        return pipeline
    except Exception as e:
        st.error(f"Error initializing pipeline: {e}")
        return None


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Multi-Modal RAG QA System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Multi-Modal RAG QA System")
    st.markdown("Upload a PDF and ask questions about it!")
    
    # Initialize session state
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = None

    # Check if index exists
    index_exists = Path("data/index/faiss_index.bin").exists()
    
    # Sidebar configuration (ALWAYS SHOW)
    with st.sidebar:
        st.header("Configuration")

        # API Key input (always show at top)
        st.subheader("API Key")
        api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.get("groq_api_key", ""),
            help="Enter your Groq API key. Get one at https://console.groq.com/",
            placeholder="gsk_...",
            key="api_key_input"
        )
        if api_key_input:
            st.session_state.groq_api_key = api_key_input
        
        # Show API key status
        if os.getenv("GROQ_API_KEY") or st.session_state.get("groq_api_key"):
            st.success("API Key configured")
        else:
            st.warning("API Key required")
            st.markdown("[Get API Key](https://console.groq.com/)")
        
        st.divider()
        
        # Upload/Reset section (ALWAYS SHOW)
        st.subheader("Document")
        if index_exists or st.session_state.pdf_processed:
            st.success("PDF processed and ready")
            if st.button("Process New PDF", use_container_width=True, type="primary"):
                # Clear index
                index_path = Path("data/index/faiss_index.bin")
                metadata_path = Path("data/index/metadata.pkl")
                if index_path.exists():
                    index_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Clear session state
                st.session_state.pdf_processed = False
                st.session_state.processing_stats = None
                if "messages" in st.session_state:
                    st.session_state.messages = []
                
                st.rerun()
        else:
            st.info("Upload a PDF above to get started")
        
        st.divider()
        
        # Show additional settings only if index exists
        if index_exists or st.session_state.pdf_processed:
            # Model selection
            llm_model = st.selectbox(
                "LLM Model",
                [
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "llama-3.2-90b-text-preview",
                ],
                index=0,
                help="Select the Groq LLM model to use",
            )
            st.session_state.llm_model = llm_model

            # Retrieval settings
            st.subheader("Retrieval Settings")
            top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)

            content_filter = st.selectbox(
                "Content Type Filter",
                ["All", "Text", "Images", "Tables"],
                index=0,
            )

            # Advanced settings
            with st.expander("Advanced Settings"):
                show_context = st.checkbox("Show retrieved context", value=False)
                show_chunks = st.checkbox("Show individual chunks", value=False)

            st.divider()

            # Pipeline stats
            if st.button("Show Pipeline Stats"):
                try:
                    pipeline = initialize_pipeline()
                    if pipeline:
                        stats = pipeline.get_pipeline_stats()
                        st.json(stats)
                except Exception as e:
                    st.error(f"Error: {e}")
            
            # Show processing stats if available
            if st.session_state.processing_stats:
                st.divider()
                st.subheader("Document Stats")
                stats = st.session_state.processing_stats
                st.metric("Total Vectors", stats.get("total_vectors", 0))
                st.metric("Pages", stats.get("pages", 0))
                st.metric("Text Chunks", stats.get("text_chunks", 0))
                st.metric("Images", stats.get("images", 0))
                st.metric("Tables", stats.get("tables", 0))
        
        # Info section (always show)
        with st.expander("About"):
            st.markdown("""
            **Multi-Modal RAG QA System**
            
            This system can answer questions about:
            - Text content
            - Tables and data
            - Images (via OCR)
            
            **How it works:**
            1. Extracts content from PDFs
            2. Generates embeddings
            3. Retrieves relevant chunks
            4. Generates answers with LLM
            
            **Tips:**
            - Be specific in your questions
            - Use content filters for targeted results
            - Check retrieved chunks to verify sources
            """)
        
        # Sample questions (always show)
        with st.expander("Sample Questions"):
            st.markdown("""
            Try asking:
            - "What is the main topic of this document?"
            - "Summarize the key findings"
            - "What data is shown in the tables?"
            - "List all the important dates mentioned"
            - "What are the conclusions?"
            """)
    
    # PDF Upload Section (show if no index exists)
    if not index_exists and not st.session_state.pdf_processed:
        st.info("Welcome! Please upload a PDF document to get started.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload PDF Document",
                type=["pdf"],
                help="Upload a PDF file to process and ask questions about"
            )
        
        with col2:
            st.markdown("### What happens next?")
            st.markdown("""
            1. Extract text, tables & images
            2. Run OCR on images
            3. Generate embeddings
            4. Build search index
            5. Chat with your document!
            """)
        
        if uploaded_file is not None:
            st.markdown("---")
            st.subheader("Processing Your PDF")
            
            # Check API key before processing
            api_key = os.getenv("GROQ_API_KEY") or st.session_state.get("groq_api_key")
            if not api_key:
                st.error("Please set your GROQ_API_KEY in the sidebar before processing!")
                st.stop()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if st.button("Process PDF", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    result = process_pdf(uploaded_file, progress_bar, status_text)
                    
                    if result["success"]:
                        st.session_state.pdf_processed = True
                        st.session_state.processing_stats = result
                        st.success("PDF processed successfully!")
                        
                        # Show stats
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Pages", result["pages"])
                        col2.metric("Text Chunks", result["text_chunks"])
                        col3.metric("Images", result["images"])
                        col4.metric("Tables", result["tables"])
                        
                        st.info("You can now start asking questions! Refresh the page or scroll down.")
                        st.balloons()
                        
                        # Force rerun to show chat interface
                        st.rerun()
                    else:
                        st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
        
        st.stop()  # Stop here if no PDF processed yet

    # Chat Interface (only show if index exists)
    if not Path("data/index/faiss_index.bin").exists():
        st.error("No PDF processed yet. Please upload a PDF first.")
        st.stop()
    
    # Set default values for variables (in case sidebar wasn't shown)
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    # Get values from sidebar or use defaults
    top_k = 5
    content_filter = "All"
    show_context = False
    show_chunks = False
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.markdown("---")
    st.subheader("Chat with Your Document")
    
    # Quick start suggestions (only show if no messages)
    if len(st.session_state.messages) == 0:
        st.markdown("### Quick Start")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Summarize Document", use_container_width=True):
                st.session_state.quick_question = "Provide a comprehensive summary of this document."
        
        with col2:
            if st.button("Key Points", use_container_width=True):
                st.session_state.quick_question = "What are the main key points and findings?"
        
        with col3:
            if st.button("Show Tables", use_container_width=True):
                st.session_state.quick_question = "What tables or data are present in the document?"
        
        st.divider()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show metadata if available
            if "metadata" in message and message["role"] == "assistant":
                with st.expander("Details"):
                    meta = message["metadata"]
                    st.write(f"**Context Summary:**")
                    st.json(meta.get("context_summary", {}))
                    st.write(f"**Tokens Used:** {meta.get('usage', {}).get('total_tokens', 'N/A')}")

    # Chat input (check for quick question first)
    question = None
    if "quick_question" in st.session_state:
        question = st.session_state.quick_question
        del st.session_state.quick_question
    else:
        question = st.chat_input("Ask a question about your document...")
    
    if question:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize pipeline
                    pipeline = initialize_pipeline()
                    
                    if pipeline is None:
                        st.error("Failed to initialize pipeline. Please check your API key.")
                        st.stop()

                    # Map content filter
                    content_type_filter = None
                    if content_filter == "Text":
                        content_type_filter = "text"
                    elif content_filter == "Images":
                        content_type_filter = "image"
                    elif content_filter == "Tables":
                        content_type_filter = "table"

                    # Query pipeline
                    result = pipeline.query(
                        question=question,
                        top_k=top_k,
                        content_type_filter=content_type_filter,
                        return_context=show_context or show_chunks,
                    )

                    # Display answer
                    st.markdown(result["answer"])

                    # Show context if requested
                    if show_context and "context" in result:
                        with st.expander("Retrieved Context"):
                            st.text(result["context"])

                    # Show individual chunks if requested
                    if show_chunks and "retrieved_chunks" in result:
                        with st.expander("Retrieved Chunks"):
                            for chunk in result["retrieved_chunks"]:
                                meta = chunk["metadata"]
                                
                                # Color code by content type
                                content_type = meta.get('content_type', 'unknown')
                                type_label = {
                                    'text': 'TEXT',
                                    'table': 'TABLE',
                                    'image': 'IMAGE'
                                }.get(content_type, 'UNKNOWN')
                                
                                st.markdown(
                                    f"**[{type_label}] Rank {chunk['rank']}** "
                                    f"(Similarity: {chunk['similarity']:.3f})"
                                )
                                st.markdown(
                                    f"*Type:* {content_type.title()}, "
                                    f"*Page:* {meta.get('page_num', '?')}"
                                )
                                
                                # Show content preview
                                content = meta.get('text', meta.get('table_text', meta.get('ocr_text', 'N/A')))
                                if len(content) > 300:
                                    st.text(content[:300] + "...")
                                else:
                                    st.text(content)
                                st.divider()

                    # Add assistant message to chat
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "metadata": {
                                "context_summary": result.get("context_summary", {}),
                                "usage": result.get("usage", {}),
                            },
                        }
                    )

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Clear chat button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
