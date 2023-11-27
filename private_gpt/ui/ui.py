"""This file should be imported only and only if you want to run the UI locally."""
import itertools
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.llms import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/bot.png"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "bot.png"
AVATAR_HUMAN = THIS_DIRECTORY_RELATIVE / "human.png"

UI_TAB_TITLE = "Korridor AlphaBot"

SOURCES_SEPARATOR = "\n\n Context sources: \n"


class Source(BaseModel):
    file: str
    page: str
    text: str
    score: float

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> set["Source"]:
        curated_sources = set()

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata
            score = chunk.score

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text, score=score)
            curated_sources.add(source)

        return curated_sources


@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service

        # Cache the UI blocks
        self._ui_block = None

    _show_sources:bool = False
    _temperature:float = 0.05

    def _chat(self, message: str, history: list[list[str]], mode: str, system_prompt: str, *_: Any) -> Any:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response

            #print(f'show_sources: {self._show_sources}')
            if self._show_sources and "I am sorry" not in full_response and completion_gen.sources and len(completion_gen.sources) > 0:
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
                sources_list = list(cur_sources)
                #sort by score descending
                sources_list.sort(key=lambda x: x.score, reverse=True)
                sources_text = "\n\n\n".join(
                    f"{index}. {source.file} (page {source.page}) - Score: {round(source.score, 4)}"
                    for index, source in enumerate(sources_list, start=1)
                )
                full_response += sources_text
            yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = list(
                itertools.chain(
                    *[
                        [
                            ChatMessage(content=interaction[0], role=MessageRole.USER),
                            ChatMessage(
                                # Remove from history content the Sources information
                                content=interaction[1].split(SOURCES_SEPARATOR)[0],
                                role=MessageRole.ASSISTANT,
                            ),
                        ]
                        for interaction in history if not interaction[1].startswith("I am sorry")
                    ]
                )
            )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        match mode:
            case "Query Docs":
                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    system_prompt=system_prompt,
                    temperature=self._temperature,
                )
                yield from yield_deltas(query_stream)

            case "LLM Chat":
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                    system_prompt=system_prompt,
                    temperature=self._temperature,
                )
                yield from yield_deltas(llm_stream)

            case "Search in Docs":
                response = self._chunks_service.retrieve_relevant(
                    text=message, limit=4, prev_next_chunks=0
                )

                sources = Source.curate_sources(response)

                yield "\n\n\n".join(
                    f"{index}. **{source.file} "
                    f"(page {source.page})**\n "
                    f"{source.text}"
                    for index, source in enumerate(sources, start=1)
                )

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            files.add(file_name)
        return [[row] for row in files]

    def _upload_file(self, files: list[str]) -> None:
        logger.debug("Loading count=%s files", len(files))
        for file in files:
            logger.info("Loading file=%s", file)
            path = Path(file)
            self._ingest_service.ingest(file_name=path.name, file_data=path)

    def _vote(data: gr.LikeData):
        print(f"data: {data}")
        if data.value:
            print("You upvoted this response: " + data.value)
        else:
            print("You downvoted this response: " + data.value)
    
    def _set_show_sources(self, checked:bool) -> None:
        self._show_sources = checked

    def _set_temperature(self, temperature:float) -> None:
        self._temperature = temperature

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.gray, neutral_hue=gr.themes.colors.stone).set(button_primary_background_fill="#00249C" ),
            css=".logo { "
            "display:flex;"
            "background-color: #00249C;"
            "height: 80px;"
            "border-radius: 8px;"
            "align-content: center;"
            "justify-content: center;"
            "align-items: center;"
            "font-size: 1.5em;"
            "}"
            ".logo img { height: 60%; padding-right: 20px }"
            ".gradio-container-3-50-2 .prose * { color: #ffffff !important;}",
        ) as blocks:
            cb=gr.Chatbot(
                    label="AlphaBot",
                    show_copy_button=True,
                    height=600,
                    render=False,
                    avatar_images=(
                        AVATAR_HUMAN,
                        AVATAR_BOT,
                    ),
                )
            
            cb.like(fn=self._vote)

            with gr.Row():
                gr.HTML(f"<div class='logo'><img src='https://korridor.com/wp-content/uploads/2021/02/Korridor-Logo-Hor-White.png'/> <div syle='padding-top:10px;color:#ffffff !important;'>AlphaBot</div></div")

            with gr.Row():
                with gr.Column(scale=3, variant="compact"):
                    mode = gr.Radio(
                        ["Query Docs", "Search in Docs", "LLM Chat"],
                        label="Mode",
                        value="Query Docs",
                    )
                    upload_button = gr.components.UploadButton(
                        "Upload File(s)",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                    )

                    ingested_dataset = gr.List(
                        self._list_ingested_files,
                        headers=["File name"],
                        label="Ingested Files",
                        interactive=False,
                        render=False,  # Rendered under the button
                    )
                    upload_button.upload(
                        self._upload_file,
                        inputs=upload_button,
                        outputs=ingested_dataset,
                    )
                    
                    ingested_dataset.change(
                        self._list_ingested_files,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()

                    show_sources = gr.Checkbox(label="Show sources in answers", value=False)
                    show_sources.change(fn=self._set_show_sources, inputs=show_sources, outputs=[])

                    temperature = gr.Slider(0, 2, value=0.05, label="Temperature", step=0.01, info="Lower values results in more deterministic responses")
                    temperature.change(fn=self._set_temperature, inputs=temperature, outputs=[])

                    system_prompt = gr.Textbox(
                (
                """You are a helpful assistant named AlphaBot and you are an expert on transporting dangerous goods in sub-Saharan Africa. You do not talk about topics that are not related to transporting dangerous goods by road in sub-Saharan Africa.
You are respectful, professional, and inclusive.
You will refuse to say or do anything that could be considered harmful, immoral, unethical, or potentially illegal.
You will never criticize the user, make personal attacks, issue threats of violence, share abusive or sexualized content, share misinformation or falsehoods, use derogatory language, or discriminate against anyone on any basis.
You will not respond to personal questions.
Always answer the query using the provided context information, and not prior knowledge. If the context is not clear just say "I am sorry but I am unable to provide an answer. Let me know if I can assist with anything else?" and do not elaborate.
Instructions and rules you must follow:
1. If you do not know the answer or the answer is not in the context, don't make one up, just say ""I am sorry but I am unable to provide an answer. Let me know if I can assist with anything else?" and do not elaborate.
2. When the user thanks you, respond with "You are welcome 🙂, anything else I can assist with?".
3. You give detailed responses but keep it concise and to the point. Avoid statements like 'It is important to note ...', 'It is also important to ...' and similar statements.
4. Never use statements like 'It is important to note ...' , 'It is also important to ...', or anything along those lines. 
5. Never use statements like 'According to the provided context, ...' , 'As per the provided context, ...', 'Based on the context, ...' or 'The context information ...' or anything along those lines. 
6. Never start a sentence with 'As per the provided context'.
7. Never start a sentence with 'Based on the provided context'."""
                ), label="System Prompt")
                    
                with gr.Column(scale=9):
                    _ = gr.ChatInterface(
                        self._chat,
                         examples=[["When can a dangerous goods inspector enter my vehicle?"],["May a dangerous goods inspector detain my vehicle?"],["Am I cool?"],["Why is the sky blue?"],
                        ["What are the steps to take during a typical daily inspection schedule?"],
                        ["What is the SIPDE driving principle?"],
                        ["What are the effects of exposure to Dangerous Goods?"],
                        ["What are important points to remember about ingestion?"]
                        ],
                        cache_examples=False,
                        chatbot=cb,
                        additional_inputs=[mode, system_prompt, upload_button, show_sources],
                    )
        return blocks
    
    


    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
