from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential
import requests


class GeminiModel:
    """Google Gemini API Provider (gemini-1.5-pro/flash)"""

    def __init__(self, model_id="gemini-1.5-pro", api_key=None):
        assert api_key is not None, "No Google API key is provided."

        # 配置 Google Generative AI
        genai.configure(api_key=api_key)
        self.model_id = model_id
        self.model = None

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(
        self,
        messages: List,
        temperature=0,
        max_tokens=5000,
    ) -> str:
        """
        将通用消息格式转换为 Gemini 格式：
        1. system 角色提取为 System Instruction
        2. assistant 角色转换为 model
        """

        system_instruction = None
        contents = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append({"role": "model", "parts": [content]})
            else:
                contents.append({"role": "user", "parts": [content]})

        # 初始化模型（带系统指令）
        self.model = genai.GenerativeModel(
            model_name=self.model_id, system_instruction=system_instruction
        )

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self.model.generate_content(
            contents, generation_config=generation_config
        )

        if not response or not response.text:
            raise ValueError("No response text returned from Gemini API.")

        return response.text
