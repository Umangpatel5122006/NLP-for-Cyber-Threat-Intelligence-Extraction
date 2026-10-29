# Phase 2 - Preprocessing

Project: NLP for Cyber Threat Intelligence Extraction

## What we did

Four small modules, each following the shared `BaseModule` contract and signed up in the registry. They run in this order (set in `configs/config.yaml`):

1. **cleaner** (`src/cti/preprocessing/cleaner.py`)
   - Normalizes Unicode with NFC (safe for Gujarati, Hindi, Marathi; NFKC can alter Indic characters)
   - Removes junk invisible characters (`\u200b`, `\ufeff`) but keeps `\u200c` / `\u200d` that Indian scripts need
   - Collapses extra whitespace; never changes real words
2. **defanger** (`defanger.py`)
   - Restores defanged indicators: `hxxps://` -> `https://`, `[.]` -> `.`, `[at]` -> `@`, `[:]` -> `:`
   - Keeps the original text in `doc.metadata["text_before_defang"]`
3. **language_detector** (`language_detector.py`)
   - Script-based guess: en, hi, mr, gu (Marathi vs Hindi by a few common Marathi words)
   - Stores result in `doc.metadata["language"]` and `["script"]`
4. **sentence_splitter** (`sentence_splitter.py`)
   - Regex splitter that does not break on dots inside IPs/domains or after "e.g."
   - Understands the Hindi/Marathi full stop (danda)
   - Stores sentences with start/end positions in `doc.metadata["sentences"]`

## Result

`config.yaml` pipeline: cleaner -> defanger -> language_detector -> sentence_splitter.
End-to-end test passed: messy defanged text came out clean, language `en`, 2 sentences.

## Swap-in alternatives (same interface)

- Sentence splitter: NLTK Punkt, spaCy, wtpsplit (neural, multilingual)
- Language detector: langdetect, lingua, fastText language-ID

## Files created

```
src/cti/preprocessing/cleaner.py
src/cti/preprocessing/defanger.py
src/cti/preprocessing/language_detector.py
src/cti/preprocessing/sentence_splitter.py
configs/config.yaml (updated)
tests/test_cleaner.py, test_defanger.py, test_language.py, test_splitter.py, test_pipeline.py
```

## Next: Phase 3 - Regex IOC extractor

IP, CVE, hash first; then URL, domain, email.
