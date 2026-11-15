"""Fixed classical normalization and label-independent concept features."""

import html
import re
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


def normalize_message(text):
    text = html.unescape(unicodedata.normalize("NFKC", text)).casefold()
    text = text.replace("\ufffd", " ")
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"(?:hxxps?|https?)://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\b[^\s@]+@[^\s@]+\b", " emailtoken ", text)
    text = re.sub(r"[₹£€$]", " currencytoken ", text)
    text = re.sub(r"\d+(?:[.,]\d+)*", " numtoken ", text)
    return " ".join(text.split())


CONCEPTS = {
    "money": r"\b(?:rupees?|rs|inr|currencytoken|cash|money|funds?|lakhs?|lacs?|crores?|refund|credit|deposit|payout|payment)\b",
    "offer": r"\b(?:giving|give|receive|claim|collect|entitled|eligible|grant|reward|prize|won|winner|free|cashback|selected|benefit|redeem)\b",
    "authority": r"\b(?:government|pm|prime minister|president|official|rbi|reserve bank|tax|department)\b",
    "account": r"\b(?:account|bank|banking|wallet|card|balance|upi|kyc)\b",
    "request": r"\b(?:reply|sms|text|send|share|forward|call|contact|click|visit|provide|enter|submit|confirm|verify|activate|pay|transfer|message)\b",
    "credential": r"\b(?:otp|password|pin|cvv|one.time code|verification code|login|credentials)\b",
    "urgency": r"\b(?:urgent|immediately|today|expire|expires|suspend|suspended|blocked|limited|final|last chance|now)\b",
    "fee": r"\b(?:fee|processing|registration|advance|charge|shipping|delivery|parcel|customs|toll)\b",
    "job": r"\b(?:job|salary|hiring|interview|recruitment|vacancy|employment)\b",
    "investment": r"\b(?:invest|investment|profit|returns|guaranteed|trading|crypto)\b",
    "warning": r"\b(?:never|do not|don't|beware|scam|fraud|fake|phishing)\b",
    "completed": r"\b(?:(?:has|have)(?: been)?|was|were|already|is) (?:credited|deposited|paid|transferred|received)\b",
    "support": r"\b(?:support|help|statement|balance|receipt|reference|transaction)\b",
}


class MessageConcepts(TransformerMixin, BaseEstimator):
    """Binary lexical cues and interactions; all decision weights are fitted."""

    @staticmethod
    def describe(text):
        text = normalize_message(text)
        values = {name: float(bool(re.search(pattern, text))) for name, pattern in CONCEPTS.items()}
        for left, right in (("money", "offer"), ("offer", "request"), ("account", "request"),
                            ("credential", "request"), ("money", "authority"), ("fee", "job"),
                            ("warning", "credential"), ("completed", "support")):
            values[left + "_and_" + right] = values[left] * values[right]
        return values

    def fit(self, texts, y=None):
        self.vectorizer_ = DictVectorizer(sparse=True)
        self.vectorizer_.fit([self.describe(t) for t in texts])
        return self

    def transform(self, texts):
        return self.vectorizer_.transform([self.describe(t) for t in texts])

    def get_feature_names_out(self, input_features=None):
        return self.vectorizer_.get_feature_names_out()
