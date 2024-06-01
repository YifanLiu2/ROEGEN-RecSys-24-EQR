import abc


class LLM(abc.ABC):
    """
    Abstract class for Language Model (LLM)
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int = 16000) -> str:
        """
        Generate text from the prompt
        :param prompt:
        :param max_tokens:
        """
        pass
