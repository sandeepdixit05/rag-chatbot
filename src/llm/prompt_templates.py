"""Prompt templates for RAG chatbot interactions."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PromptTemplate:
    """Represents a prompt template with variables."""

    template: str
    required_variables: List[str]
    optional_variables: List[str] = None
    description: str = ""

    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        # Check required variables
        missing = [var for var in self.required_variables if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Set defaults for optional variables
        if self.optional_variables:
            for var in self.optional_variables:
                if var not in kwargs:
                    kwargs[var] = ""

        return self.template.format(**kwargs)


class PromptTemplates:
    """Collection of prompt templates for different use cases."""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates."""
        return {
            "rag_answer": PromptTemplate(
                template="""You are a helpful AI assistant that answers questions based on provided context from PDF documents.

Context Information:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context above
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite specific parts of the context when possible
4. Be concise but comprehensive in your response
5. If there are multiple relevant pieces of information, synthesize them coherently

Answer:""",
                required_variables=["context", "question"],
                description="Main RAG answering template"
            ),

            "rag_answer_with_sources": PromptTemplate(
                template="""You are a helpful AI assistant that answers questions based on provided context from PDF documents.

Context Information:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context above
2. Include source references (page numbers, document names) when available
3. If the context doesn't contain enough information to answer the question, say so clearly
4. Be concise but comprehensive in your response
5. Format your answer with clear source citations

Answer:""",
                required_variables=["context", "question"],
                description="RAG answering template with source citations"
            ),

            "conversation_rag": PromptTemplate(
                template="""You are a helpful AI assistant having a conversation about PDF documents. Use the provided context to answer questions and maintain conversation flow.

Previous Conversation:
{conversation_history}

Current Context from Documents:
{context}

Current Question: {question}

Instructions:
1. Consider the conversation history to maintain context and continuity
2. Answer the current question using the provided document context
3. Reference previous parts of the conversation when relevant
4. Be conversational while remaining accurate to the source material
5. If clarification is needed, ask follow-up questions

Answer:""",
                required_variables=["conversation_history", "context", "question"],
                description="Conversational RAG template with memory"
            ),

            "summarization": PromptTemplate(
                template="""Please provide a comprehensive summary of the following text from a PDF document.

Text to Summarize:
{text}

Instructions:
1. Create a concise but comprehensive summary
2. Preserve key information and main points
3. Maintain the logical flow of ideas
4. Include important details and examples when relevant
5. Keep the summary to approximately {max_length} words

Summary:""",
                required_variables=["text"],
                optional_variables=["max_length"],
                description="Text summarization template"
            ),

            "question_classification": PromptTemplate(
                template="""Classify the following question to help determine the best retrieval strategy.

Question: {question}

Classification Categories:
1. FACTUAL - Asking for specific facts, numbers, dates, definitions
2. PROCEDURAL - Asking how to do something or about processes
3. CONCEPTUAL - Asking about concepts, theories, or explanations
4. COMPARATIVE - Comparing different items, options, or approaches
5. ANALYTICAL - Requiring analysis, evaluation, or synthesis
6. CONTEXTUAL - Requiring understanding of specific context or situation

Instructions:
Respond with just the category name (e.g., "FACTUAL") followed by a brief explanation.

Classification:""",
                required_variables=["question"],
                description="Question classification template"
            ),

            "context_evaluation": PromptTemplate(
                template="""Evaluate whether the provided context is sufficient to answer the given question.

Question: {question}

Context:
{context}

Instructions:
Rate the context sufficiency on a scale of 1-5:
1 - No relevant information
2 - Minimal relevant information
3 - Some relevant information but gaps exist
4 - Most information needed is present
5 - Comprehensive information available

Provide the rating followed by a brief explanation.

Evaluation:""",
                required_variables=["question", "context"],
                description="Context sufficiency evaluation template"
            ),

            "follow_up_questions": PromptTemplate(
                template="""Based on the question and answer provided, suggest relevant follow-up questions that a user might want to ask.

Original Question: {question}
Answer: {answer}
Available Context Topics: {context_topics}

Instructions:
1. Generate 3-5 relevant follow-up questions
2. Consider what additional information the user might need
3. Base suggestions on the available context topics
4. Make questions specific and actionable

Follow-up Questions:""",
                required_variables=["question", "answer"],
                optional_variables=["context_topics"],
                description="Follow-up question generation template"
            ),

            "clarification_needed": PromptTemplate(
                template="""The user's question might be ambiguous or could benefit from clarification to provide a better answer.

Question: {question}
Available Context: {context_summary}

Instructions:
1. Identify what aspects of the question could be clarified
2. Suggest specific clarifying questions
3. Keep clarifications relevant to available context
4. Be helpful and conversational

Clarification Request:""",
                required_variables=["question"],
                optional_variables=["context_summary"],
                description="Clarification request template"
            ),

            "no_context_response": PromptTemplate(
                template="""You are a helpful AI assistant. The user has asked a question, but no relevant context was found in the available PDF documents.

Question: {question}

Instructions:
1. Politely explain that you don't have information about this topic in the available documents
2. Suggest how the user might rephrase their question or what they could look for
3. Be helpful and encouraging
4. Don't make up information

Response:""",
                required_variables=["question"],
                description="Template for when no relevant context is found"
            ),

            "system_message": PromptTemplate(
                template="""You are an AI assistant specialized in answering questions based on PDF documents. Your capabilities include:

- Analyzing and understanding content from PDF documents
- Providing accurate answers based on the available context
- Citing sources when possible
- Asking clarifying questions when needed
- Maintaining conversation context

Guidelines:
- Only use information from the provided context
- Be accurate and honest about limitations
- Cite sources when available
- Ask for clarification when questions are ambiguous

Current session context: {session_context}""",
                required_variables=[],
                optional_variables=["session_context"],
                description="System message template"
            ),

            "error_handling": PromptTemplate(
                template="""I apologize, but I encountered an issue while processing your request.

Error Type: {error_type}
User Question: {question}

I'm having trouble {error_description}. Here are some suggestions:

1. Try rephrasing your question
2. Be more specific about what you're looking for
3. Check if your question relates to the available documents

Please try again with a different approach.

{additional_help}""",
                required_variables=["error_type", "question", "error_description"],
                optional_variables=["additional_help"],
                description="Error handling template"
            )
        }

    def get_template(self, name: str) -> PromptTemplate:
        """Get a specific template by name."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found. Available: {list(self.templates.keys())}")
        return self.templates[name]

    def format_template(self, name: str, **kwargs) -> str:
        """Format a specific template with provided variables."""
        template = self.get_template(name)
        return template.format(**kwargs)

    def list_templates(self) -> Dict[str, str]:
        """List all available templates with descriptions."""
        return {name: template.description for name, template in self.templates.items()}

    def create_context_string(
        self,
        retrieval_results: List[Any],
        include_sources: bool = True,
        max_context_length: Optional[int] = None
    ) -> str:
        """Create formatted context string from retrieval results."""
        if not retrieval_results:
            return "No relevant context found."

        context_parts = []
        current_length = 0

        for i, result in enumerate(retrieval_results, 1):
            # Format source information
            source_info = ""
            if include_sources and hasattr(result, 'source_info') and result.source_info:
                source_parts = []
                if result.source_info.get('file_name'):
                    source_parts.append(f"Document: {result.source_info['file_name']}")
                if result.source_info.get('page_number'):
                    source_parts.append(f"Page: {result.source_info['page_number']}")

                if source_parts:
                    source_info = f" ({', '.join(source_parts)})"

            # Format context entry
            context_entry = f"[Context {i}]{source_info}:\n{result.content}\n"

            # Check length limit
            if max_context_length and current_length + len(context_entry) > max_context_length:
                context_parts.append(f"\n[Note: Additional context available but truncated for length]")
                break

            context_parts.append(context_entry)
            current_length += len(context_entry)

        return "\n".join(context_parts)

    def create_conversation_history(
        self,
        messages: List[Dict[str, str]],
        max_history_length: Optional[int] = None
    ) -> str:
        """Create formatted conversation history string."""
        if not messages:
            return "No previous conversation."

        # Limit history length if specified
        if max_history_length:
            messages = messages[-max_history_length:]

        history_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'user':
                history_parts.append(f"User: {content}")
            elif role == 'assistant':
                history_parts.append(f"Assistant: {content}")

        return "\n".join(history_parts)

    def extract_context_topics(self, retrieval_results: List[Any]) -> str:
        """Extract main topics from retrieval results for follow-up suggestions."""
        if not retrieval_results:
            return "General topics from available documents"

        # Simple topic extraction from metadata and content
        topics = set()

        for result in retrieval_results:
            # Add document names as topics
            if hasattr(result, 'source_info') and result.source_info:
                file_name = result.source_info.get('file_name', '')
                if file_name:
                    # Extract topic from filename (remove extension, replace underscores)
                    topic = file_name.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
                    topics.add(topic)

            # Add metadata topics if available
            if hasattr(result, 'metadata'):
                title = result.metadata.get('title', '')
                subject = result.metadata.get('subject', '')
                if title:
                    topics.add(title)
                if subject:
                    topics.add(subject)

        return ", ".join(sorted(topics)) if topics else "Available document content"