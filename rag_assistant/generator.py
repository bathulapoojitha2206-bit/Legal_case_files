"""
Generation module for grounded legal question answering.
Supports Ollama (local default) and optional cloud providers (OpenAI, Groq, Mock).
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rag_assistant.retriever import RetrievedPassage

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when the Ollama local daemon is unavailable."""
    pass


class OllamaModelNotFoundError(Exception):
    """Raised when the requested model is not downloaded in Ollama."""
    pass


@dataclass
class GenerationResult:
    """Encapsulates the final generated response and metadata."""
    answer: str
    model: str
    provider: str
    prompt: str
    context_used: str
    passages: List[RetrievedPassage]


class LegalResponseGenerator:
    """
    Generates grounded answers based strictly on retrieved legal passages.
    Prevents hallucination through strict system prompt constraints.
    """

    DEFAULT_PROMPT_TEMPLATE = """You are an expert Legal Document Knowledge Assistant.
Your role is to analyze legal case documents and answer questions strictly based on the provided retrieved context passages.

STRICT OPERATIONAL GUIDELINES:
1. Base your answer EXCLUSIVELY on the factual information contained in the Context below.
2. If the answer cannot be determined or is not mentioned in the Context, you MUST clearly state:
   "The requested information is not available in the retrieved case documents."
3. Do NOT extrapolate, speculate, invent facts, or create citations that are not in the text.
4. Maintain a clear, factual, and professional tone suitable for legal research.
5. Reference relevant dates, monetary amounts, case numbers, or clauses when present in the context.

--- CONTEXT PASSAGES ---
{context}
--- END OF CONTEXT ---

USER QUESTION:
{question}

ANSWER:"""

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        request_timeout: float = 60.0,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.2-3b-preview",
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model

    def check_ollama_status(self) -> Tuple[bool, str, List[str]]:
        """
        Check if Ollama server is alive and list available models.

        Returns:
            Tuple of (is_connected, status_message, list_of_model_names).
        """
        url = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LegalRAGAssistant/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return True, "Connected to Ollama daemon.", models
        except urllib.error.URLError as err:
            return (
                False,
                f"Cannot connect to Ollama at {self.base_url}. Ensure 'ollama serve' is running.",
                []
            )
        except Exception as exc:
            return False, f"Ollama connection check failed: {exc}", []

    def format_context(self, passages: List[RetrievedPassage]) -> str:
        """Format retrieved passages into a readable context string with source citations."""
        if not passages:
            return "No relevant passages were found in the indexed documents."

        formatted_blocks = []
        for p in passages:
            page_info = f", Page {p.page}" if p.page else ""
            chunk_info = f", Chunk #{p.chunk_id}" if p.chunk_id is not None else ""
            header = f"[Source: {p.filename}{page_info}{chunk_info} | Match Score: {p.score:.3f}]"
            formatted_blocks.append(f"{header}\n{p.content.strip()}")

        return "\n\n".join(formatted_blocks)

    def generate(
        self,
        question: str,
        passages: List[RetrievedPassage],
        custom_prompt_template: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate a grounded answer for the question using retrieved passages.

        Args:
            question: The user query.
            passages: Retrieved document passages.
            custom_prompt_template: Optional custom prompt template.

        Returns:
            GenerationResult object.
        """
        question_clean = question.strip()
        if not question_clean:
            return GenerationResult(
                answer="Please provide a valid, non-empty legal question.",
                model=self.model_name,
                provider=self.provider,
                prompt="",
                context_used="",
                passages=[]
            )

        context_str = self.format_context(passages)
        template = custom_prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        prompt = template.format(context=context_str, question=question_clean)

        if not passages:
            # If no passages could be retrieved, avoid calling LLM and return grounded notice
            return GenerationResult(
                answer="The requested information is not available in the retrieved case documents.",
                model=self.model_name,
                provider=self.provider,
                prompt=prompt,
                context_used=context_str,
                passages=[]
            )

        if self.provider == "ollama":
            answer = self._generate_ollama(prompt)
            model_used = self.model_name
        elif self.provider == "openai":
            answer = self._generate_openai(prompt)
            model_used = self.openai_model
        elif self.provider == "groq":
            answer = self._generate_groq(prompt)
            model_used = self.groq_model
        elif self.provider == "mock":
            answer = self._generate_mock(question_clean, passages)
            model_used = "mock-llm"
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        return GenerationResult(
            answer=answer,
            model=model_used,
            provider=self.provider,
            prompt=prompt,
            context_used=context_str,
            passages=passages
        )

    def _generate_ollama(self, prompt: str) -> str:
        """Execute generation via local Ollama API."""
        # 1. Verify daemon connectivity
        is_alive, msg, installed_models = self.check_ollama_status()
        if not is_alive:
            raise OllamaConnectionError(
                f"Ollama server is unavailable ({msg}). "
                "Please run 'ollama serve' in your terminal or start the Ollama application."
            )

        # 2. Check if requested model exists
        base_model_name = self.model_name.split(":")[0]
        model_exists = any(
            self.model_name == m or base_model_name == m.split(":")[0]
            for m in installed_models
        )
        if not model_exists and installed_models:
            raise OllamaModelNotFoundError(
                f"Model '{self.model_name}' is not downloaded in Ollama. "
                f"Available models: {', '.join(installed_models)}. "
                f"To install, run: 'ollama pull {self.model_name}' in your terminal."
            )

        # 3. Call Ollama chat API
        try:
            import ollama
            client = ollama.Client(host=self.base_url)
            response = client.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.temperature,
                }
            )
            return response.get("message", {}).get("content", "").strip()
        except Exception as exc:
            logger.error("Ollama generation error: %s", exc, exc_info=True)
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

    def _generate_openai(self, prompt: str) -> str:
        """Optional cloud fallback using OpenAI."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured in the environment.")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            completion = client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    def _generate_groq(self, prompt: str) -> str:
        """Optional cloud fallback using Groq."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured in the environment.")
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)
            completion = client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            raise RuntimeError(f"Groq API call failed: {exc}") from exc

    def _generate_mock(self, question: str, passages: List[RetrievedPassage]) -> str:
        """Deterministic mock generator for fast CI unit tests without running Ollama."""
        if not passages:
            return "The requested information is not available in the retrieved case documents."
        first_passage = passages[0].content[:200].replace("\n", " ")
        return f"[Grounded Mock Answer based on {passages[0].filename}]: {first_passage}..."
