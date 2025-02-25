import os
import base64
from dataclasses import dataclass, field


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def format_message_image_openai(prompt, base64_image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ]


class Model:
    def __init__(self, api_name, max_tokens=6000, **api_kwargs):
        self.api_name = api_name
        self.max_tokens = max_tokens
        self.api_kwargs = api_kwargs
        self.init_client()

    def init_client(self):
        raise NotImplementedError("Model must implement init_client method.")

    def predict(self, prompt, temperature=0.5):
        raise NotImplementedError("Model must implement predict method.")

    def predict_image(self, prompt, base64_image, temperature=0.5):
        raise NotImplementedError("Model must implement predict method.")


class BaseCompletionStyleModel(Model):
    """A base class for models that use the OpenAI completion API."""

    def predict(self, prompt, temperature=0.5):
        completion = self.client.chat.completions.create(
            model=self.api_name,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content
        return response

    def predict_image(self, prompt, base64_image, temperature=0.5):

        completion = self.client.chat.completions.create(
            model=self.api_name,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=format_message_image_openai(prompt, base64_image),
        )
        response = completion.choices[0].message.content
        return response


class OpenAIModel(Model):
    def init_client(self):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def predict(self, prompt, temperature=0.5):
        from openai import NOT_GIVEN

        completion = self.client.chat.completions.create(
            model=self.api_name,
            max_tokens=(self.max_tokens if self.max_tokens is not None else NOT_GIVEN),
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=(
                self.api_kwargs["max_completion_tokens"]
                if "max_completion_tokens" in self.api_kwargs
                else NOT_GIVEN
            ),
            reasoning_effort=(
                self.api_kwargs["reasoning_effort"]
                if "reasoning_effort" in self.api_kwargs
                else NOT_GIVEN
            ),
        )
        response = completion.choices[0].message.content
        return response

    def predict_image(self, prompt, base64_image, temperature=0.5):
        from openai import NOT_GIVEN

        completion = self.client.chat.completions.create(
            model=self.api_name,
            max_tokens=(self.max_tokens if self.max_tokens is not None else NOT_GIVEN),
            temperature=temperature,
            messages=format_message_image_openai(prompt, base64_image),
            max_completion_tokens=(
                self.api_kwargs["max_completion_tokens"]
                if "max_completion_tokens" in self.api_kwargs
                else NOT_GIVEN
            ),
            reasoning_effort=(
                self.api_kwargs["reasoning_effort"]
                if "reasoning_effort" in self.api_kwargs
                else NOT_GIVEN
            ),
        )
        response = completion.choices[0].message.content
        return response


class XAIModel(BaseCompletionStyleModel):
    def init_client(self):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )


class OpenRouterModel(BaseCompletionStyleModel):
    def init_client(self):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )


class DeepInfraModel(BaseCompletionStyleModel):
    def init_client(self):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )


class AnthropicModel(Model):
    def init_client(self):
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def predict(self, prompt, temperature=0.5):
        completion = self.client.messages.create(
            model=self.api_name,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.content[0].text
        return response

    def predict_image(self, prompt, base64_image, temperature=0.5):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        completion = self.client.messages.create(
            model=self.api_name,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=messages,
        )
        response = completion.content[0].text
        return response


class AnthropicThinkingModel(Model):
    def init_client(self):
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def predict(self, prompt, temperature=None):
        completion = self.client.messages.create(
            model=self.api_name,
            max_tokens=self.max_tokens,
            thinking={
                "type": "enabled",
                "budget_tokens": self.api_kwargs["budget_tokens"]
            },
            messages=[{"role": "user", "content": prompt}],
        )
        # response = completion.content[-1].text
        response = ''
        for block in completion.content:
            if block.type == 'text':
                response += block.text
            elif block.type == 'thinking':
                response += '\n<thinking>\n' + block.thinking + '\n</thinking>\n'
        return response

class AzureOpenAIModel(Model):
    def init_client(self):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        )

    def predict(self, prompt, temperature=0.5):
        from openai import NOT_GIVEN

        completion = self.client.chat.completions.create(
            model=self.api_name,
            max_tokens=(self.max_tokens if self.max_tokens is not None else NOT_GIVEN),
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=(
                self.api_kwargs["max_completion_tokens"]
                if "max_completion_tokens" in self.api_kwargs
                else NOT_GIVEN
            ),
            reasoning_effort=(
                self.api_kwargs["reasoning_effort"]
                if "reasoning_effort" in self.api_kwargs
                else NOT_GIVEN
            ),
        )
        response = completion.choices[0].message.content

        completion_tokens = completion.usage.completion_tokens

        return response


class MistralModel(Model):
    def init_client(self):
        from mistralai import Mistral

        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def predict(self, prompt, temperature=0.5):
        completion = self.client.chat.complete(
            model=self.api_name,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content
        return response


class GoogleModel(Model):
    def init_client(self):
        # from google import genai
        import google.generativeai as genai

        # self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(self.api_name)

    def predict(self, prompt, temperature=0.5):
        import google.generativeai as genai
        from google.api_core import retry
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=self.max_tokens,
        )
        completion = self.model.generate_content(
            contents=prompt, generation_config=generation_config,
            request_options={"retry": retry.Retry(maximum=4),}
        )
        response = completion.text
        return response


class DeepseekModel(Model):
    def init_client(self):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )

    def predict(self, prompt, temperature=0.5):
        completion = self.client.chat.completions.create(
            model=self.api_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        cot_content = completion.choices[0].message.reasoning_content
        content = completion.choices[0].message.content
        response = "<cot>\n" + cot_content + "\n</cot>\n\n" + content

        return response


class ModelEngineFactory:
    """
    Dynamically create or retrieve a model engine based on the model name.
    The factory ensures we only initialize the model client once per model_name
    (if you wish).
    """

    _model_engines = {}  # cache of already created engines

    # The default temperature is 0.5, but we can override it for specific models
    _temperature_dict = {
        "o1-preview-2024-09-12": 1,
        "o1-mini": 1,
        "o1-2024-12-17-high": 1,
        "o1-2024-12-17-med": 1,
        "o3-mini-2025-01-31-high": 1,
        "claude-3-7-sonnet-20250219-thinking": None,
    }

    reasoning_models = [
        "o1-preview-2024-09-12",
        "o1-mini",
        "o1-2024-12-17-high",
        "o1-2024-12-17-med",
        "o3-mini-2025-01-31-high",
        "deepseek-r1",
        "gemini-2.0-flash-thinking",
        "gemini-2.0-flash-thinking-01-21",
        "claude-3-7-sonnet-20250219-thinking",
    ]

    @classmethod
    def get_engine(cls, model_name):
        # If we already created an engine for this general "family", return it
        # or you can choose to cache them by exact `model_name`.

        if model_name in cls._model_engines:
            return cls._model_engines[model_name]

        if model_name == "gpt-4o-2024-08-06":
            engine = OpenAIModel(api_name="gpt-4o-2024-08-06")
        elif model_name == "gpt-4o-2024-11-20":
            engine = OpenAIModel(api_name="gpt-4o-2024-11-20")
        elif model_name == "gpt-4o-mini":
            engine = OpenAIModel(api_name="gpt-4o-mini-2024-07-18")
        elif model_name == "o1-preview-2024-09-12":
            engine = OpenAIModel(
                api_name="o1-preview-2024-09-12",
                max_completion_tokens=10000,
                max_tokens=None,
            )
        elif model_name == "o1-mini":
            engine = OpenAIModel(
                api_name="o1-mini",
                max_completion_tokens=10000,
                max_tokens=None,
            )
        elif model_name == "claude-3-5-sonnet":
            engine = AnthropicModel(api_name="claude-3-5-sonnet-20240620")
        elif model_name == "claude-3-5-sonnet-20241022":
            engine = AnthropicModel(api_name="claude-3-5-sonnet-20241022")
        elif model_name == "claude-3-7-sonnet-20250219":
            engine = AnthropicModel(api_name="claude-3-7-sonnet-20250219", max_tokens=16000)
        elif model_name == "claude-3-7-sonnet-20250219-thinking":
            engine = AnthropicThinkingModel(api_name="claude-3-7-sonnet-20250219", max_tokens=20000, budget_tokens=16000)
        elif model_name == "claude-3-5-haiku":
            engine = AnthropicModel(api_name="claude-3-5-haiku-20241022")
        elif model_name == "mistral-large":
            engine = MistralModel(api_name="mistral-large-2407")
        elif model_name == "mistral-small":
            engine = MistralModel(api_name="mistral-small-2409")
        elif model_name == "gemini-1.5-flash":
            engine = GoogleModel(api_name="gemini-1.5-flash-002")
        elif model_name == "gemini-1.5-pro":
            engine = GoogleModel(api_name="gemini-1.5-pro-002")
        elif model_name == "gemini-2.0-flash":
            engine = GoogleModel(api_name="gemini-2.0-flash-001")
        elif model_name == "gemini-2.0-flash-thinking":
            engine = GoogleModel(
                api_name="gemini-2.0-flash-thinking-exp-1219",
                max_completion_tokens=32000,
            )
        elif model_name == "gemini-2.0-flash-thinking-01-21":
            engine = GoogleModel(
                api_name="gemini-2.0-flash-thinking-exp-01-21",
                max_completion_tokens=32000,
            )
        elif model_name == "gemini-2.0-pro-02-05":
            engine = GoogleModel(api_name="gemini-2.0-pro-exp-02-05")
        elif model_name == "grok-2-1212":
            engine = XAIModel(api_name="grok-2-1212")
        elif model_name == "grok-2-vision-1212":
            engine = XAIModel(api_name="grok-2-vision-1212")
        elif model_name == "deepseek/deepseek-chat":
            engine = OpenRouterModel(api_name="deepseek/deepseek-chat")
        elif model_name == "meta-llama/Meta-Llama-3.1-405B-Instruct":
            engine = DeepInfraModel(api_name="meta-llama/Meta-Llama-3.1-405B-Instruct")
        elif model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct":
            engine = DeepInfraModel(api_name="meta-llama/Meta-Llama-3.1-70B-Instruct")
        elif model_name == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            engine = DeepInfraModel(
                api_name="meta-llama/Meta-Llama-3.2-11B-Vision-Instruct"
            )
        elif model_name == "meta-llama/Llama-3.2-90B-Vision-Instruct":
            engine = DeepInfraModel(
                api_name="meta-llama/Meta-Llama-3.2-90B-Vision-Instruct"
            )
        elif model_name == "meta-llama/Llama-3.3-70B-Instruct":
            engine = DeepInfraModel(api_name="meta-llama/Llama-3.3-70B-Instruct")
        elif model_name == "Qwen/Qwen2.5-72B-Instruct":
            engine = DeepInfraModel(api_name="Qwen/Qwen2.5-72B-Instruct")
        elif model_name == "Qwen2.5-Max":
            engine = OpenRouterModel(api_name="qwen/qwen-max")
        elif model_name == "o1-2024-12-17-med":
            engine = OpenAIModel(
                api_name="o1-2024-12-17",
                reasoning_effort="medium",
                max_tokens=None,  # depracated
                max_completion_tokens=32000,
            )
        elif model_name == "o1-2024-12-17-high":
            engine = OpenAIModel(
                api_name="o1-2024-12-17",
                reasoning_effort="high",
                max_tokens=None,  # depracated
                max_completion_tokens=32000,
            )
        elif model_name == "deepseek-r1":
            engine = DeepseekModel(api_name="deepseek-reasoner", max_tokens=8192)
        elif model_name == "o3-mini-2025-01-31-high":
            engine = OpenAIModel(
                api_name="o3-mini-2025-01-31",
                max_completion_tokens=32000,
                max_tokens=None,  # depracated
                reasoning_effort="high",
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")

        cls._model_engines[model_name] = engine
        return engine

    @classmethod
    def get_temperature(cls, model_name):
        if model_name in cls._temperature_dict:
            return cls._temperature_dict[model_name]
        else:
            return 0.5


class ModelInferenceEngine:
    """
    This class manages:
    - The response cache
    - The building of prompts into the appropriate "messages" structure
    - The retrieval of the correct model engine from the factory
    - The caching logic (force refresh, etc.)
    """

    def __init__(self, response_cache):
        self.response_cache = response_cache

    def set_response_cache(self, response_cache):
        self.response_cache = response_cache

    def run_inference(
        self,
        prompt,
        model_name,
        image_path=None,
        run_id=0,
        force_refresh=False,
        load_only=False,
    ):
        """
        Run inference on a single prompt. If cached, returns from the cache.
        """

        temperature = ModelEngineFactory.get_temperature(model_name)

        if image_path is not None:
            key = (
                prompt,
                os.path.basename(image_path),
                temperature,
                run_id,
                model_name,
            )
        else:
            key = (prompt, temperature, run_id, model_name)

        # If only loading from cache, ensure the key exists
        if load_only and not self.response_cache.has(key):
            raise ValueError(f"Prompt not found in cache: {prompt}")

        # If the cache has it and not forcing a refresh, return
        if self.response_cache.has(key) and not force_refresh:
            output = self.response_cache.get(key)
            if output is None or output == "":
                print(
                    f"A cached response for {model_name} was empty, which likely indicates an API error. Rerunning..."
                )
            else:
                return key, output, 0

        # Get the model engine from the factory
        engine = ModelEngineFactory.get_engine(model_name)

        if image_path is not None:
            base64_image = encode_image(image_path)
            response = engine.predict_image(
                prompt,
                base64_image,
                temperature=temperature,
            )

        else:
            response = engine.predict(
                prompt,
                temperature=temperature,
            )

        # Save to cache
        self.response_cache.set(key, response)
        return key, response, 1
