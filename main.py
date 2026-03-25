"""Main entry point for the RAG Chatbot application."""

import os
import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import settings
from src.chatbot import RAGChatbot
from src.chatbot.chat_interface import StreamlitInterface, GradioInterface, FastAPIInterface

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)

logger = logging.getLogger(__name__)


def create_chatbot_config() -> dict:
    """Create chatbot configuration from settings."""
    return {
        # API Keys
        'openai_api_key': settings.openai_api_key,
        'anthropic_api_key': settings.anthropic_api_key,

        # Model Settings
        'embedding_model': settings.embedding_model,
        'llm_provider': settings.llm_provider,
        'llm_model': settings.llm_model,

        # Vector Database
        'vector_db_path': settings.vector_db_path,
        'collection_name': settings.collection_name,

        # Document Processing
        'chunk_size': settings.chunk_size,
        'chunk_overlap': settings.chunk_overlap,
        'max_retrieval_docs': settings.max_retrieval_docs,

        # PDF Processing
        'pdf_input_dir': settings.pdf_input_dir,
        'processed_dir': settings.processed_dir,

        # LLM Settings
        'temperature': settings.temperature,
        'max_tokens': settings.max_tokens
    }


def run_streamlit():
    """Run Streamlit interface."""
    try:
        config = create_chatbot_config()
        chatbot = RAGChatbot(config)

        # Load documents if PDF directory exists and has files
        pdf_dir = Path(settings.pdf_input_dir)
        if pdf_dir.exists() and list(pdf_dir.glob("*.pdf")):
            logger.info(f"Loading documents from {pdf_dir}")
            result = chatbot.load_documents(pdf_dir)
            if result.get("success"):
                logger.info(f"Loaded {result['pdfs_processed']} PDFs successfully")
            else:
                logger.warning(f"Document loading failed: {result.get('error')}")

        interface = StreamlitInterface(chatbot, "📚 RAG Chatbot")
        interface.run()

    except Exception as e:
        logger.error(f"Error running Streamlit interface: {str(e)}")
        raise


def run_gradio(share=False, port=7860):
    """Run Gradio interface."""
    try:
        config = create_chatbot_config()
        chatbot = RAGChatbot(config)

        # Load documents if available
        pdf_dir = Path(settings.pdf_input_dir)
        if pdf_dir.exists() and list(pdf_dir.glob("*.pdf")):
            logger.info(f"Loading documents from {pdf_dir}")
            result = chatbot.load_documents(pdf_dir)
            if result.get("success"):
                logger.info(f"Loaded {result['pdfs_processed']} PDFs successfully")

        interface = GradioInterface(chatbot, "🤖 RAG Chatbot")
        interface.run(share=share, server_port=port)

    except Exception as e:
        logger.error(f"Error running Gradio interface: {str(e)}")
        raise


def run_api(host="127.0.0.1", port=8000):
    """Run FastAPI interface."""
    try:
        config = create_chatbot_config()
        chatbot = RAGChatbot(config)

        # Load documents if available
        pdf_dir = Path(settings.pdf_input_dir)
        if pdf_dir.exists() and list(pdf_dir.glob("*.pdf")):
            logger.info(f"Loading documents from {pdf_dir}")
            result = chatbot.load_documents(pdf_dir)
            if result.get("success"):
                logger.info(f"Loaded {result['pdfs_processed']} PDFs successfully")

        interface = FastAPIInterface(chatbot, "RAG Chatbot API")
        logger.info(f"Starting API server at http://{host}:{port}")
        interface.run(host=host, port=port)

    except Exception as e:
        logger.error(f"Error running API interface: {str(e)}")
        raise


def run_cli():
    """Run command-line interface."""
    try:
        config = create_chatbot_config()
        chatbot = RAGChatbot(config)

        # Load documents if available
        pdf_dir = Path(settings.pdf_input_dir)
        if pdf_dir.exists() and list(pdf_dir.glob("*.pdf")):
            print(f"Loading documents from {pdf_dir}...")
            result = chatbot.load_documents(pdf_dir)
            if result.get("success"):
                print(f"✅ Loaded {result['pdfs_processed']} PDFs successfully")
                print(f"Created {result['embedded_chunks']} chunks")
            else:
                print(f"❌ Document loading failed: {result.get('error')}")
                return
        else:
            print(f"No PDF files found in {pdf_dir}")
            print("Please add PDF files to the directory and restart.")
            return

        print("\n🤖 RAG Chatbot CLI")
        print("Type 'quit' or 'exit' to stop, 'help' for commands")
        print("-" * 50)

        session_id = None

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help    - Show this help message")
                    print("  stats   - Show system statistics")
                    print("  clear   - Start a new conversation")
                    print("  quit    - Exit the chatbot")
                    continue

                elif user_input.lower() == 'stats':
                    stats = chatbot.get_system_stats()
                    print(f"\nSystem Statistics:")
                    print(f"  Documents: {stats.get('vector_store', {}).get('total_documents', 0)}")
                    print(f"  Active Sessions: {stats.get('conversation_manager', {}).get('active_sessions', 0)}")
                    print(f"  Model: {stats.get('llm_model', {}).get('model', 'Unknown')}")
                    continue

                elif user_input.lower() == 'clear':
                    session_id = None
                    print("🔄 Started new conversation")
                    continue

                elif not user_input:
                    continue

                # Process the message
                print("\n🤖 Assistant: ", end="", flush=True)

                response = chatbot.chat(
                    message=user_input,
                    session_id=session_id,
                    include_sources=True
                )

                if response.get("error"):
                    print(f"❌ Error: {response['error']}")
                    continue

                session_id = response.get("session_id")
                print(response["response"])

                # Show sources if available
                if response.get("sources"):
                    print(f"\n📚 Sources ({len(response['sources'])}):")
                    for i, source in enumerate(response["sources"], 1):
                        doc_name = source.get('document', 'Unknown')
                        page = source.get('page')
                        page_info = f" (Page {page})" if page else ""
                        print(f"  {i}. {doc_name}{page_info}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                logger.error(f"CLI error: {str(e)}")

    except Exception as e:
        logger.error(f"Error running CLI interface: {str(e)}")
        print(f"❌ Failed to start chatbot: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Chatbot - Chat with your PDF documents")
    parser.add_argument(
        "interface",
        choices=["streamlit", "gradio", "api", "cli"],
        help="Interface to run"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address (for API mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number (for API/Gradio mode)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public sharing link (Gradio mode)"
    )
    parser.add_argument(
        "--load-docs",
        help="Path to directory containing PDF files to load"
    )

    args = parser.parse_args()

    # Update PDF input directory if provided
    if args.load_docs:
        settings.pdf_input_dir = args.load_docs

    # Validate configuration
    if settings.llm_provider in ["openai", "anthropic"]:
        api_key = settings.openai_api_key if settings.llm_provider == "openai" else settings.anthropic_api_key
        if not api_key:
            print(f"❌ Error: {settings.llm_provider.upper()}_API_KEY environment variable is required")
            print("Please set your API key in the .env file or environment variables")
            return

    # Create necessary directories
    Path(settings.pdf_input_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.vector_db_path).mkdir(parents=True, exist_ok=True)
    Path(settings.processed_dir).mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting RAG Chatbot ({args.interface} interface)")
    print(f"📁 PDF Directory: {settings.pdf_input_dir}")
    print(f"🧠 LLM Provider: {settings.llm_provider} ({settings.llm_model})")
    print(f"🔍 Embedding Model: {settings.embedding_model}")

    try:
        if args.interface == "streamlit":
            run_streamlit()
        elif args.interface == "gradio":
            run_gradio(share=args.share, port=args.port)
        elif args.interface == "api":
            run_api(host=args.host, port=args.port)
        elif args.interface == "cli":
            run_cli()

    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"❌ Application failed: {str(e)}")


if __name__ == "__main__":
    main()