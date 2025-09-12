import openai
from typing import Dict, Any, List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Pipeline RAG: Busca contexto relevante, monta prompt, consulta LLM e retorna resposta e fontes.
    """

    def __init__(self, db_manager, embedding_generator, api_key: str, model_slug: str = "openai/gpt-3.5-turbo"):
        self.db_manager = db_manager
        self.embedding_generator = embedding_generator
        self.model_slug = model_slug
        self.api_key = api_key
        # Configura o endpoint OpenRouter (OpenAI compatível)
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_type = "openai"
        openai.api_version = None  # Não é necessário para OpenRouter

    def generate_answer(self, question: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Executa o pipeline RAG: busca contexto, monta prompt, consulta LLM e retorna resposta e fontes.

        Args:
            question (str): Pergunta do usuário.
            max_results (int): Número de documentos de contexto a buscar.

        Returns:
            Dict[str, Any]: Resposta gerada, fontes, modelo e uso de tokens.
        """
        try:
            # 1. Gerar embedding da pergunta
            logger.info("Gerando embedding da pergunta.")
            question_embedding = self.embedding_generator.generate_embeddings([question])[0]

            # 2. Buscar documentos relevantes
            logger.info("Buscando documentos relevantes no ChromaDB.")
            search_results = self.db_manager.search(question_embedding, n_results=max_results)
            docs = search_results.get("documents", [[]])[0]
            metadatas = search_results.get("metadatas", [[]])[0]

            # 3. Construir contexto
            context = "\n\n".join(docs)
            if not context:
                logger.warning("Nenhum contexto relevante encontrado.")
                return {
                    "answer": "Nenhum contexto relevante encontrado para responder à pergunta.",
                    "sources": [],
                    "model_used": self.model_slug,
                    "tokens_used": 0
                }

            # 4. Montar prompt para o LLM
            prompt = (
                f"Contexto:\n{context}\n\n"
                f"Pergunta: {question}\n"
                f"Responda de forma clara e cite as fontes relevantes se possível."
            )

            # 5. Chamar o modelo via OpenRouter (API OpenAI compatível)
            openai.api_key = self.api_key
            logger.info("Chamando o modelo LLM via OpenRouter.")
            response = openai.ChatCompletion.create(
                model=self.model_slug,
                messages=[
                    {"role": "system", "content": "Você é um assistente útil que responde perguntas com base no contexto fornecido."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.2,
            )

            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if hasattr(response, "usage") else 0

            # 6. Retornar resposta, fontes e uso de tokens
            return {
                "answer": answer,
                "sources": [
                    {"content": doc, "metadata": meta}
                    for doc, meta in zip(docs, metadatas)
                ],
                "model_used": self.model_slug,
                "tokens_used": tokens_used
            }

        except Exception as oe:
            logger.error(f"Erro na chamada ao modelo LLM: {oe}")
            return {
                "answer": "Erro ao consultar o modelo LLM. Verifique sua chave de API e limite de uso.",
                "sources": [],
                "model_used": self.model_slug,
                "tokens_used": 0
            }
        except Exception as e:
            logger.error(f"Erro inesperado no pipeline RAG: {e}")
            return {
                "answer": "Erro inesperado ao gerar resposta.",
                "sources": [],
                "model_used": self.model_slug,
                "tokens_used": 0
            }