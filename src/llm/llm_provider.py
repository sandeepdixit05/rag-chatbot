"""LLM provider implementations for various AI services."""

import logging
from typing import Dict, List, Optional, Union, Any, Iterator
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""

    content: str
    model: str
    usage: Dict[str, int]
    response_time: float
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class StreamingLLMResponse:
    """Represents a streaming response from an LLM."""

    content_stream: Iterator[str]
    model: str
    metadata: Dict[str, Any] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_streaming_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> StreamingLLMResponse:
        """Generate a streaming response from the LLM."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            base_url: Base URL for API (for custom endpoints)
        """
        if openai is None:
            raise ImportError("openai package not installed. Install with: pip install openai")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url

        # Initialize client
        if base_url:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=api_key)

        logger.info(f"Initialized OpenAI provider with model: {model}")

    def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()

        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            response_time = time.time() - start_time

            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                response_time=response_time,
                finish_reason=response.choices[0].finish_reason,
                metadata={"openai_response": response}
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    def generate_streaming_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> StreamingLLMResponse:
        """Generate streaming response using OpenAI API."""
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            def content_generator():
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            return StreamingLLMResponse(
                content_stream=content_generator(),
                model=self.model,
                metadata={"provider": "openai"}
            )

        except Exception as e:
            logger.error(f"OpenAI streaming API error: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "base_url": self.base_url,
            "supports_streaming": True,
            "supports_system_message": True
        }


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307"
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model name to use
        """
        if anthropic is None:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")

        self.api_key = api_key
        self.model = model

        # Initialize client
        self.client = anthropic.Anthropic(api_key=api_key)

        logger.info(f"Initialized Anthropic provider with model: {model}")

    def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()

        try:
            # Anthropic uses different parameter structure
            request_params = {
                "model": self.model,
                "max_tokens": max_tokens or 2000,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }

            if system_message:
                request_params["system"] = system_message

            # Add any additional kwargs
            request_params.update(kwargs)

            response = self.client.messages.create(**request_params)

            response_time = time.time() - start_time

            return LLMResponse(
                content=response.content[0].text,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                response_time=response_time,
                finish_reason=response.stop_reason,
                metadata={"anthropic_response": response}
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise

    def generate_streaming_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> StreamingLLMResponse:
        """Generate streaming response using Anthropic API."""
        try:
            request_params = {
                "model": self.model,
                "max_tokens": max_tokens or 2000,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }

            if system_message:
                request_params["system"] = system_message

            request_params.update(kwargs)

            stream = self.client.messages.create(**request_params)

            def content_generator():
                for chunk in stream:
                    if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                        yield chunk.delta.text

            return StreamingLLMResponse(
                content_stream=content_generator(),
                model=self.model,
                metadata={"provider": "anthropic"}
            )

        except Exception as e:
            logger.error(f"Anthropic streaming API error: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        return {
            "provider": "anthropic",
            "model": self.model,
            "supports_streaming": True,
            "supports_system_message": True
        }


class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider (placeholder for local models like Ollama)."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """Initialize local LLM provider.

        Args:
            model: Local model name
            base_url: Base URL for local LLM server
            **kwargs: Additional configuration
        """
        self.model = model
        self.base_url = base_url
        self.config = kwargs

        logger.info(f"Initialized local LLM provider with model: {model}")

    def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using local LLM."""
        # This is a placeholder implementation
        # In a real implementation, you would integrate with Ollama, llama.cpp, or similar

        start_time = time.time()

        # Placeholder response
        response_content = f"[Local LLM Response] Based on the provided context, here's my response to: {prompt[:100]}..."

        response_time = time.time() - start_time

        return LLMResponse(
            content=response_content,
            model=self.model,
            usage={
                "prompt_tokens": len(prompt) // 4,  # Rough estimate
                "completion_tokens": len(response_content) // 4,
                "total_tokens": (len(prompt) + len(response_content)) // 4
            },
            response_time=response_time,
            metadata={"provider": "local", "base_url": self.base_url}
        )

    def generate_streaming_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> StreamingLLMResponse:
        """Generate streaming response using local LLM."""
        def content_generator():
            # Placeholder streaming implementation
            response = f"[Local LLM Streaming] Response to: {prompt[:50]}..."
            for word in response.split():
                yield word + " "
                time.sleep(0.05)  # Simulate streaming delay

        return StreamingLLMResponse(
            content_stream=content_generator(),
            model=self.model,
            metadata={"provider": "local", "base_url": self.base_url}
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get local model information."""
        return {
            "provider": "local",
            "model": self.model,
            "base_url": self.base_url,
            "supports_streaming": True,
            "supports_system_message": True,
            "config": self.config
        }


class LLMProvider:
    """Factory class for creating LLM providers."""

    @staticmethod
    def create_provider(
        provider_type: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """Create an LLM provider instance.

        Args:
            provider_type: Type of provider ("openai", "anthropic", "local")
            api_key: API key for cloud providers
            model: Model name to use
            **kwargs: Additional provider-specific arguments

        Returns:
            LLM provider instance
        """
        provider_type = provider_type.lower()

        if provider_type == "openai":
            if not api_key:
                raise ValueError("API key required for OpenAI provider")

            model = model or "gpt-4o-mini"
            return OpenAIProvider(
                api_key=api_key,
                model=model,
                **kwargs
            )

        elif provider_type == "anthropic":
            if not api_key:
                raise ValueError("API key required for Anthropic provider")

            model = model or "claude-3-haiku-20240307"
            return AnthropicProvider(
                api_key=api_key,
                model=model,
                **kwargs
            )

        elif provider_type == "local":
            model = model or "llama2"
            return LocalLLMProvider(
                model=model,
                **kwargs
            )

        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

    @staticmethod
    def get_supported_providers() -> List[str]:
        """Get list of supported provider types."""
        return ["openai", "anthropic", "local"]

    @staticmethod
    def validate_provider_config(
        provider_type: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Validate provider configuration."""
        provider_type = provider_type.lower()
        validation_result = {
            "valid": False,
            "provider": provider_type,
            "requirements": [],
            "warnings": []
        }

        if provider_type == "openai":
            validation_result["requirements"].append("OpenAI API key")
            if openai is None:
                validation_result["warnings"].append("openai package not installed")
            if api_key:
                validation_result["valid"] = True

        elif provider_type == "anthropic":
            validation_result["requirements"].append("Anthropic API key")
            if anthropic is None:
                validation_result["warnings"].append("anthropic package not installed")
            if api_key:
                validation_result["valid"] = True

        elif provider_type == "local":
            validation_result["requirements"].append("Local LLM server running")
            validation_result["valid"] = True  # Assume local setup is valid

        else:
            validation_result["warnings"].append(f"Unknown provider: {provider_type}")

        return validation_result