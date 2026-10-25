from abc import ABC, abstractmethod


class BaseTagger(ABC):
    """Contract every NER model must follow."""
    name = "base"

    @abstractmethod
    def fit(self, tokens, tags, dev_tokens=None, dev_tags=None):
        """tokens: list of token lists. tags: list of BIO tag lists."""

    @abstractmethod
    def predict(self, tokens):
        """Return a list of BIO tag lists, one per sentence."""