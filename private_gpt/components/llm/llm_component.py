from injector import inject, singleton
from llama_index.llms import MockLLM
from llama_index.llms.base import LLM
from llama_index.llms.llama_utils import completion_to_prompt, messages_to_prompt

from private_gpt.paths import models_path
from private_gpt.settings.settings import Settings


@singleton
class LLMComponent:
    llm: LLM

    @inject
    def __init__(self, settings: Settings) -> None:
        match settings.llm.mode:
            case "local":
                from llama_index.llms import LlamaCPP

                self.llm = LlamaCPP(
                    model_path=str(models_path / settings.local.llm_hf_model_file),
                    temperature=0.5,
                    # llama2 has a context window of 4096 tokens,
                    # but we set it lower to allow for some wiggle room
                    context_window=3900,
                    generate_kwargs={},
                    # All to GPU
                    model_kwargs={"n_gpu_layers": -1, "top_p": 0.5, "max_new_tokens": 2000},
                    # transform inputs into Llama2 format
                    messages_to_prompt=messages_to_prompt,
                    completion_to_prompt=completion_to_prompt,
                    verbose=True,
                )

            case "sagemaker":
                from private_gpt.components.llm.custom.sagemaker import SagemakerLLM

                self.llm = SagemakerLLM(
                    endpoint_name=settings.sagemaker.llm_endpoint_name,
                )
            case "openai":
                print(f"Using OpenAI model: {settings.openai.model}")
                from llama_index.llms import OpenAI

                openai_key = settings.openai.api_key
                self.llm = OpenAI(
                    api_key=openai_key, 
                    model=settings.openai.model,
                    temperature=0.05, 
                    verbose=True,)
                
                from llama_index.llms import OpenAI
            case "mock":
                self.llm = MockLLM()
