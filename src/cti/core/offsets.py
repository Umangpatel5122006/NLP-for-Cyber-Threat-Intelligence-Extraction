"""Track character provenance through the project's text normalization steps.

Each current character carries a covering interval in the supplied report.
Intervals may overlap after Unicode composition, expansion, or reordering.
Positions use Python Unicode code points and half-open intervals, not bytes.
"""

from dataclasses import dataclass
import unicodedata


def _canonical_units(text):
    """NFD units with their input-character positions, canonically ordered."""
    units, marks = [], []
    for index, character in enumerate(text):
        for unit in unicodedata.normalize("NFD", character):
            if unicodedata.combining(unit) == 0:
                units.extend(sorted(marks, key=lambda item: unicodedata.combining(item[0])))
                marks.clear()
                units.append((unit, index))
            else:
                marks.append((unit, index))
    units.extend(sorted(marks, key=lambda item: unicodedata.combining(item[0])))
    return units


@dataclass(frozen=True)
class MappedText:
    original: str
    text: str
    origins: tuple[tuple[int, int], ...]

    @classmethod
    def from_text(cls, text):
        if not isinstance(text, str):
            raise TypeError("Document.text must be a string.")
        return cls(text, text, tuple((index, index + 1) for index in range(len(text))))

    def original_span(self, start, end):
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.text):
            raise ValueError("Evidence offsets must be a nonempty half-open span within document text.")
        origins = self.origins[start:end]
        return min(span[0] for span in origins), max(span[1] for span in origins)

    def normalize_nfc(self):
        normalized = unicodedata.normalize("NFC", self.text)
        if normalized == self.text:
            return self
        # Canonical decompositions are equal, even when NFC absorbs a mark
        # across another mark or composes Hangul. Align NFD units, not an
        # approximate string diff or separately normalized input characters.
        before, after = _canonical_units(self.text), _canonical_units(normalized)
        if [unit for unit, _ in before] != [unit for unit, _ in after]:
            raise ValueError("Unicode normalization changed canonical decomposition.")
        origins = [None] * len(normalized)
        for (_, source), (_, target) in zip(before, after):
            start, end = self.origins[source]
            previous = origins[target]
            origins[target] = (start, end) if previous is None else (
                min(start, previous[0]), max(end, previous[1])
            )
        return MappedText(self.original, normalized, tuple(origins))

    def sub(self, pattern, replacement):
        """Regex substitution; changed output covers the complete matched source.

        Insertions without an input span are excluded. Existing cleaner and
        defanger rules consume nonempty spans, including deleted characters.
        """
        chunks, origins, position = [], [], 0
        for match in pattern.finditer(self.text):
            start, end = match.span()
            if start == end:
                raise ValueError("Offset tracking does not support zero-width substitutions.")
            chunks.append(self.text[position:start])
            origins.extend(self.origins[position:start])
            value = match.expand(replacement)
            chunks.append(value)
            if value == match.group():
                origins.extend(self.origins[start:end])
            elif value:
                origins.extend([self.original_span(start, end)] * len(value))
            position = end
        if position == 0:
            return self
        chunks.append(self.text[position:])
        origins.extend(self.origins[position:])
        return MappedText(self.original, "".join(chunks), tuple(origins))

    def strip(self):
        stripped = self.text.strip()
        if stripped == self.text:
            return self
        start = len(self.text) - len(self.text.lstrip())
        end = start + len(stripped)
        return MappedText(self.original, stripped, self.origins[start:end])
