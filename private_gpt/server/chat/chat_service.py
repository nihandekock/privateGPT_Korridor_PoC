from injector import inject, singleton
from llama_index import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.chat_engine import ContextChatEngine
from llama_index.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.llm_predictor.utils import stream_chat_response_to_tokens
from llama_index.llms import ChatMessage, MessageRole
from llama_index.types import TokenGen
from pydantic import BaseModel

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk


class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None


class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None


@singleton
class ChatService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm_service = llm_component
        self.vector_store_component = vector_store_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.service_context = ServiceContext.from_defaults(
            llm=llm_component.llm, embed_model=embedding_component.embedding_model
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store_component.vector_store,
            storage_context=self.storage_context,
            service_context=self.service_context,
            show_progress=True,
        )

    def _chat_engine(
        self, context_filter: ContextFilter | None = None, system_prompt: str = "", temperature: float = 0.05,
    ) -> BaseChatEngine:
        vector_index_retriever = self.vector_store_component.get_retriever(
            index=self.index, context_filter=context_filter
        )
        #print(f"temperature: {temperature}")
        self.llm_service.llm.temperature = temperature #TODO: this does not work, possibly because the llm is a singleton

        CONTEXT_TEMPLATE = (
                f"{system_prompt}\n"
                "Context information is below. Only use the context information to answer questions and do not use prior information.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                #"Given the context information and not prior knowledge, "
                #f"answer the question: {message}\n"
            )
        
        prefix_messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)] 

        #@NdK: Different models give different results when you use system_prompt or prefix_messages, we need to expirement with this.
        
        return ContextChatEngine.from_defaults(
            retriever=vector_index_retriever,
            service_context=self.service_context,
            system_prompt=system_prompt,
            #prefix_messages=prefix_messages,
            #context_template=CONTEXT_TEMPLATE,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window"),
            ],
        )

    def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
        system_prompt: str = "",
        temperature: float = 0.05,
    ) -> CompletionGen:
        if use_context:
            
            last_message = messages[-1].content
            chat_engine = self._chat_engine(
                context_filter=context_filter, system_prompt= system_prompt, temperature=temperature)
            streaming_response = chat_engine.stream_chat(
                message=last_message if last_message is not None else "",
                chat_history=messages[:-1],
            )
            sources = [
                Chunk.from_node(node) for node in streaming_response.source_nodes
            ]
            completion_gen = CompletionGen(
                response=streaming_response.response_gen, sources=sources
            )
        else:
            stream = self.llm_service.llm.stream_chat(messages)
            completion_gen = CompletionGen(
                response=stream_chat_response_to_tokens(stream)
            )
        return completion_gen

    def chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
        system_prompt: str = "",
        temperature: float = 0.05,
    ) -> Completion:
        if use_context:
            last_message = messages[-1].content
            chat_engine = self._chat_engine(
                context_filter=context_filter, system_prompt= system_prompt, temperature=temperature)
            wrapped_response = chat_engine.chat(
                message=last_message if last_message is not None else "",
                chat_history=messages[:-1],
            )
            sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
            completion = Completion(response=wrapped_response.response, sources=sources)
        else:
            chat_response = self.llm_service.llm.chat(messages)
            response_content = chat_response.message.content
            response = response_content if response_content is not None else ""
            completion = Completion(response=response)
        return completion
