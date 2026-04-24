from typing import List
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


class OpenAIModel:
    """Standard OpenAI API (api.openai.com)"""

    def __init__(self, model_id="gpt-4", api_key=None, azure_endpoint=None):
        assert api_key is not None, "No API key is provided."
        assert azure_endpoint is not None, "No Azure endpoint is provided."

        self.model_id = model_id
        self.client = AzureOpenAI(
            api_version="2025-12-01-preview",
            api_key=api_key,
            azure_endpoint=azure_endpoint,
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(
        self,
        messages: List,
        temperature=0,
        presence_penalty=0,
        frequency_penalty=0,
        max_tokens=5000,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )
        if (
            not response
            or not hasattr(response, "choices")
            or len(response.choices) == 0
        ):
            raise ValueError("No response choices returned from the API.")
        return response.choices[0].message.content
