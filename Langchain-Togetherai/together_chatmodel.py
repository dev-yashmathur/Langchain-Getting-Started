"""Together ai chat completions wrapper."""
import requests
from typing import Any, List, Optional, Dict

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs.generation import Generation
from langchain_core.outputs.run_info import RunInfo
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
    LLMResult,
)
from langchain_core.language_models.llms import LLM



DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""


class ChatTogether(BaseChatModel):
    """
    """

    llm: LLM
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None
    model_id: str = None  # type: ignore

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm.base_url = "https://api.together.xyz/v1/chat/completions"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_output = self.llm._call(
            prompts=None, messages=llm_input, stop=stop, run_manager=run_manager, **kwargs
        )
        llm_result = self._parse_to_LLMResult(llm_output)
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)

        llm_result = await self.llm._agenerate(
            prompts="", messages=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a list of chat_ml messages"""
        if not messages:
            raise ValueError("at least one HumanMessage must be provided")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("last message must be a HumanMessage")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        # The chat-completions endpoint accepts the chat_ml formatted messages list
        return messages_dicts

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )

        """Resolve the model_id from the LLM's inference_server_url"""

        from huggingface_hub import list_inference_endpoints

        available_endpoints = list_inference_endpoints("*")

        if isinstance(self.llm, HuggingFaceTextGenInference):
            endpoint_url = self.llm.inference_server_url

        elif isinstance(self.llm, HuggingFaceEndpoint):
            endpoint_url = self.llm.endpoint_url

        elif isinstance(self.llm, HuggingFaceHub):
            # no need to look up model_id for HuggingFaceHub LLM
            self.model_id = self.llm.repo_id
            return

        else:
            raise ValueError(f"Unknown LLM type: {type(self.llm)}")

        for endpoint in available_endpoints:
            if endpoint.url == endpoint_url:
                self.model_id = endpoint.repository

        if not self.model_id:
            raise ValueError(
                "Failed to resolve model_id"
                f"Could not find model id for inference server provided: {endpoint_url}"
                "Make sure that your Hugging Face token has access to the endpoint."
            )

    @property
    def _llm_type(self) -> str:
        return "togetherai-chat-wrapper"
    
    def _parse_to_LLMResult(self, llm_output) -> LLMResult:
        generations = [[Generation(text=choice["message"]["content"]) for choice in llm_output["choices"]]]

        # Optionally include LLM specific output and run information
        llm_specific_output = {
            "id": llm_output["id"],
            "usage": llm_output["usage"],
            "created": llm_output["created"],
            "model": llm_output["model"],
            "object": llm_output["object"]
        }

        # Assuming no explicit run information is given, we construct a placeholder
        # run_info = [RunInfo(run_id=llm_output['id'])]

        # Create the LLMResult object
        llm_result = LLMResult(generations=generations, llm_output=llm_specific_output,)
        return llm_result