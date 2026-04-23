from typing import List
import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential


class ClaudeModel:
    """Standard Anthropic API (Claude 3.5 Sonnet / 3 Opus)"""

    def __init__(self, model_id="claude-3-5-sonnet-20240620", api_key=None):
        assert api_key is not None, "No Anthropic API key is provided."

        # 初始化 Anthropic 客户端
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(
        self,
        messages: List,
        temperature=0,
        max_tokens=5000,
    ) -> str:
        """
        Claude 的消息格式与 OpenAI 兼容，但 system prompt 通常作为顶级参数提取
        """

        # 分离 System Message (Claude 的 API 习惯将 system 放在顶级参数中)
        system_content = ""
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                filtered_messages.append(msg)

        # 调用 Claude API
        response = self.client.messages.create(
            model=self.model_id,
            system=system_content,
            messages=filtered_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response or not response.content:
            raise ValueError("No response content returned from Claude API.")

        # Claude 返回的是内容块列表，通常取第一个文本块
        return response.content[0].text
