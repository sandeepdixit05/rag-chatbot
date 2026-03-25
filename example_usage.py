"""Example usage of the RAG Chatbot system."""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv()

from config.settings import settings
from src.chatbot import RAGChatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_config():
    """Create a sample configuration for the chatbot."""
    return {
        # API Keys (you need to set these in your .env file)
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),

        # Model Settings
        'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
        'llm_provider': 'openai',  # or 'anthropic' or 'local'
        'llm_model': 'gpt-4o-mini',

        # Vector Database Settings
        'vector_db_path': './data/vector_db',
        'collection_name': 'example_documents',

        # Document Processing
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'max_retrieval_docs': 5,

        # PDF Processing
        'pdf_input_dir': './data/pdfs',
        'processed_dir': './data/processed',

        # LLM Settings
        'temperature': 0.7,
        'max_tokens': 2000
    }


def example_basic_usage():
    """Example of basic RAG chatbot usage."""
    print("🤖 RAG Chatbot - Basic Usage Example")
    print("=" * 50)

    # Create configuration
    config = create_sample_config()

    # Initialize chatbot
    print("1. Initializing RAG Chatbot...")
    chatbot = RAGChatbot(config)

    # Check if we have an API key
    if not config.get('openai_api_key') and not config.get('anthropic_api_key'):
        print("❌ Error: No API key found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file")
        return

    print("✅ Chatbot initialized successfully!")

    # Load documents (if PDF directory exists)
    pdf_dir = Path(config['pdf_input_dir'])
    pdf_dir.mkdir(parents=True, exist_ok=True)

    if list(pdf_dir.glob("*.pdf")):
        print(f"\n2. Loading documents from {pdf_dir}...")
        result = chatbot.load_documents(pdf_dir)

        if result.get('success'):
            print(f"✅ Successfully processed {result['pdfs_processed']} PDFs")
            print(f"📄 Created {result['embedded_chunks']} text chunks")
            print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
        else:
            print(f"❌ Error loading documents: {result.get('error')}")
            return
    else:
        print(f"\n⚠️  No PDF files found in {pdf_dir}")
        print("Please add some PDF files to test the chatbot functionality.")
        print("For now, we'll demonstrate the system without documents.")

    # Example chat interactions
    print("\n3. Example Chat Interactions")
    print("-" * 30)

    if chatbot.ready_for_chat:
        # Example questions about documents
        questions = [
            "What are the main topics covered in the documents?",
            "Can you summarize the key points?",
            "What is the most important information?"
        ]
    else:
        # Questions that work without documents
        questions = [
            "Hello, how do you work?",
            "What can you help me with?",
            "Tell me about your capabilities."
        ]

    session_id = None

    for i, question in enumerate(questions, 1):
        print(f"\n👤 Question {i}: {question}")

        response = chatbot.chat(
            message=question,
            session_id=session_id,
            include_sources=True
        )

        session_id = response.get('session_id')

        if response.get('error'):
            print(f"❌ Error: {response['error']}")
        else:
            print(f"🤖 Answer: {response['response']}")

            # Show sources if available
            if response.get('sources'):
                print(f"\n📚 Sources ({len(response['sources'])}):")
                for j, source in enumerate(response['sources'], 1):
                    doc_name = source.get('document', 'Unknown')
                    page = source.get('page')
                    score = source.get('score', 0)
                    page_info = f" (Page {page})" if page else ""
                    print(f"  {j}. {doc_name}{page_info} - Relevance: {score:.2f}")

    # System statistics
    print("\n4. System Statistics")
    print("-" * 20)
    stats = chatbot.get_system_stats()

    if stats.get('error'):
        print(f"❌ Error getting stats: {stats['error']}")
    else:
        print(f"📊 Initialization Status: {stats['initialized']}")
        print(f"📄 Documents Loaded: {stats['documents_loaded']}")
        print(f"💬 Ready for Chat: {stats['ready_for_chat']}")

        if stats.get('vector_store'):
            vs_stats = stats['vector_store']
            print(f"🗃️  Total Documents in DB: {vs_stats.get('total_documents', 0)}")

        if stats.get('conversation_manager'):
            conv_stats = stats['conversation_manager']
            print(f"💭 Active Sessions: {conv_stats.get('active_sessions', 0)}")
            print(f"💭 Total Messages: {conv_stats.get('total_messages', 0)}")

    print("\n✨ Example completed!")


def example_advanced_features():
    """Example of advanced RAG chatbot features."""
    print("\n🚀 RAG Chatbot - Advanced Features Example")
    print("=" * 55)

    config = create_sample_config()
    chatbot = RAGChatbot(config)

    # Health check
    print("1. Health Check")
    health = chatbot.health_check()
    print(f"Overall Health: {health['overall']}")

    for component, status in health.get('components', {}).items():
        health_emoji = "✅" if status.get('status') == 'healthy' else "❌"
        print(f"  {health_emoji} {component}: {status.get('status', 'unknown')}")

    # Session management
    print("\n2. Session Management")
    print("Creating multiple sessions...")

    session1 = chatbot.conversation_manager.create_session(metadata={"user": "Alice"})
    session2 = chatbot.conversation_manager.create_session(metadata={"user": "Bob"})

    print(f"Created sessions: {session1}, {session2}")

    # Add messages to different sessions
    chatbot.conversation_manager.add_message(session1, "user", "Hello from Alice!")
    chatbot.conversation_manager.add_message(session2, "user", "Hello from Bob!")

    active_sessions = chatbot.list_active_sessions()
    print(f"Active sessions: {len(active_sessions)}")

    # Get session history
    history = chatbot.get_session_history(session1)
    if not history.get('error'):
        print(f"Session {session1} has {len(history['messages'])} messages")

    # Document processing pipeline demonstration
    print("\n3. Document Processing Pipeline")

    # Show each component in the pipeline
    print("Pipeline components:")
    print("  📄 PDF Extractor: Extracts text from PDF files")
    print("  ✂️  Document Processor: Splits text into chunks")
    print("  🔢 Embedding Generator: Creates vector embeddings")
    print("  🗃️  Vector Store: Stores and indexes embeddings")
    print("  🔍 Document Retriever: Finds relevant content")
    print("  🧠 LLM Provider: Generates responses")
    print("  💭 Conversation Manager: Manages chat history")

    # Retrieval system demonstration
    if chatbot.ready_for_chat:
        print("\n4. Retrieval System Test")
        test_query = "artificial intelligence"

        # Get retrieval results directly
        retrieval_results = chatbot.document_retriever.retrieve(
            query=test_query,
            top_k=3,
            rerank_query=True
        )

        print(f"Query: '{test_query}'")
        print(f"Found {len(retrieval_results)} relevant chunks:")

        for i, result in enumerate(retrieval_results, 1):
            print(f"  {i}. Score: {result.score:.3f} | {result.content[:100]}...")

    # Configuration options
    print("\n5. Configuration Options")
    print("Key configuration parameters:")
    print(f"  🔧 Chunk Size: {config['chunk_size']}")
    print(f"  🔧 Chunk Overlap: {config['chunk_overlap']}")
    print(f"  🔧 Max Retrieval Docs: {config['max_retrieval_docs']}")
    print(f"  🔧 LLM Provider: {config['llm_provider']}")
    print(f"  🔧 LLM Model: {config['llm_model']}")
    print(f"  🔧 Temperature: {config['temperature']}")

    print("\n✨ Advanced features example completed!")


def example_performance_testing():
    """Example of performance testing and optimization."""
    print("\n⚡ RAG Chatbot - Performance Testing")
    print("=" * 45)

    config = create_sample_config()
    chatbot = RAGChatbot(config)

    if not chatbot.ready_for_chat:
        print("⚠️  Skipping performance tests - no documents loaded")
        return

    import time

    # Test query performance
    print("1. Query Performance Test")
    test_queries = [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "What are the benefits of automation?",
        "Explain deep learning concepts",
        "What is natural language processing?"
    ]

    total_time = 0
    successful_queries = 0

    for i, query in enumerate(test_queries, 1):
        print(f"Testing query {i}/5: '{query[:30]}...'")

        start_time = time.time()
        try:
            response = chatbot.chat(query, include_sources=False)
            end_time = time.time()

            if not response.get('error'):
                query_time = end_time - start_time
                total_time += query_time
                successful_queries += 1
                print(f"  ✅ Response time: {query_time:.2f}s")

                # Show token usage if available
                usage = response.get('usage', {})
                if usage:
                    print(f"  📊 Tokens: {usage.get('total_tokens', 'N/A')}")
            else:
                print(f"  ❌ Error: {response['error']}")

        except Exception as e:
            print(f"  ❌ Exception: {str(e)}")

    if successful_queries > 0:
        avg_time = total_time / successful_queries
        print(f"\n📈 Performance Summary:")
        print(f"  Successful queries: {successful_queries}/{len(test_queries)}")
        print(f"  Average response time: {avg_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")

    # Memory usage (basic estimation)
    print("\n2. System Resource Usage")
    stats = chatbot.get_system_stats()

    if stats.get('vector_store'):
        vs_stats = stats['vector_store']
        print(f"  📊 Documents in vector DB: {vs_stats.get('total_documents', 0)}")

    if stats.get('conversation_manager'):
        conv_stats = stats['conversation_manager']
        print(f"  💭 Active conversations: {conv_stats.get('active_sessions', 0)}")

    print("\n⚡ Performance testing completed!")


def main():
    """Run all examples."""
    try:
        # Basic usage example
        example_basic_usage()

        # Advanced features example
        example_advanced_features()

        # Performance testing example
        example_performance_testing()

        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Add your PDF documents to ./data/pdfs/")
        print("2. Set your API keys in .env file")
        print("3. Run: python main.py streamlit")
        print("4. Or run: python main.py cli")

    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()