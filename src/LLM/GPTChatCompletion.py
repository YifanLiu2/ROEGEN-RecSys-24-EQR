from openai import OpenAI
from src.LLM.LLM import LLM

class GPTChatCompletion(LLM):
    """
    GPT Chat Completion using OpenAI API
    """

    def __init__(self, model_name: str = "gpt-4o", api_key: str = "API_KEY"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)

    def generate(self, message: list[dict], max_tokens: int = 4000, temperature: float = 0.0, json: bool = True) -> str | None:
        kwargs = {}
        if json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            print(e)
            return None
        return response.choices[0].message.content
