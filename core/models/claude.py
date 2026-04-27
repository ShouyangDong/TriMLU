from typing import List
import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential


class ClaudeModel:
    """Standard Claude API (api.anthropic.com)"""

    def __init__(self, model_id="claude-sonnet-4-5-20250929", api_key=None):
        assert api_key is not None, "No API key provided."
        self.model_id = model_id
        print("model: ", self.model_id)
        print("api_key: ", api_key)
        client_kwargs = {"api_key": api_key}
        client_kwargs["base_url"] = "https://api.ezclaude.com"
        self.client = anthropic.Anthropic(**client_kwargs)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(
        self,
        messages: List,
        temperature=0,
        presence_penalty=0,
        frequency_penalty=0,
        max_tokens=16000,
    ) -> str:
        max_tokens = min(max_tokens, 16000)
        api_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = self.client.messages.create(**api_kwargs)
        except anthropic.APIStatusError as e:
            raise ValueError(f"Anthropic API error: {e.status_code} - {e.message}")
        except anthropic.APIConnectionError as e:
            raise ValueError(f"API connection error: {str(e)}")
        except anthropic.AuthenticationError as e:
            raise ValueError(f"Authentication error - check your API key: {str(e)}")
        except Exception as e:
            raise ValueError(f"API call failed: {type(e).__name__}: {str(e)}")

        if (
            not response
            or not hasattr(response, "content")
            or len(response.content) == 0
        ):
            raise ValueError("No response content returned from the API.")
        return response.content[0].text
