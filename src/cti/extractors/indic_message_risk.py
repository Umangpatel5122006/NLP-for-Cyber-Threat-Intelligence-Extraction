"""Classical Hindi/Hinglish lexical scam checks; no translation or learned model.

Concept combinations generalize over names, amounts, targets and word order.
These are authored rules, not calibrated fraud predictions or a Hindi NER model.
"""

import re
from bisect import bisect_left

from cti.extractors.message_risk import PATTERNS as ENGLISH, SIGNALS, _is_request


INDIC_RULE_VERSION = 2
LANGUAGES = ("en", "hi", "hi-en")


def phrase(value):
    # Python \b alone splits Hindi vowel signs from their base letters.
    # SMS writers commonly omit the nasal mark on these imperative endings.
    value = value.replace("करें", "करे[ंँ]?").replace("भेजें", "भेजे[ंँ]?")
    letters = r"[\w\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962-\u0963]"
    return re.compile(r"(?<!" + letters + r")(?:" + value + r")(?!" + letters + r")", re.I)


PATTERNS = {
    "reward": phrase(r"मिलेंगे|मिलेगा|मिलेगी|पाएंगे|पाएँगे|पाएं|पाएँ|पाने|जीत(?:े|ा|ी|ें)|जीत गए|मुफ्त|मुफ़्त|इनाम|पुरस्कार|अनुदान|कैशबैक|बोनस|प्राप्त करें|दिए जाएंगे|दिये जायेंगे|दे रही|दे रहे|देगी|milega|milenge|milegi|milne|paane|pane|jeet(?:a|e|i)?|jeet gaye|inaam|inam|muft|mile hain|mil rahe|de rahi|de rahe|diye jayenge"),
    "money": phrase(r"रुपये|रुपए|रुपया|पैसे|पैसा|राशि|लाख|करोड़|करोड|नकद|धन|इनाम|लॉटरी|paise|paisa|rupaye|rupay|rupaiye|rashi|laakh|nakad|inaam|inam"),
    "bank": phrase(r"बैंक(?:\s+(?:खाते|खाता))?|खाते|खाता|भुगतान|क्रेडिट|जमा|खाता संख्या|khate|khata|khaate|khaata|bhugtan|jama"),
    "contact": phrase(r"भेजें|भेजो|भेजिए|भेजिये|बताएं|बताएँ|बताओ|बताइए|बताइये|दें|देना|भरें|भरो|(?:साझा|शेयर|क्लिक|कॉल|फोन|संपर्क|मैसेज|एसएमएस|sms|reply|click|call|share|send|pay|transfer|जमा|भुगतान|दर्ज|अपडेट)\s*(?:करें|करो|कीजिए|कीजिये)|(?:उत्तर|जवाब)\s*दें|bhejo|bheje|bhejen|bhejein|bhejiye|batao|bataye|batayen|batayein|bataiye|(?:sampark|samparka|share|saajha|sajha|sms|call|phone|reply|click|pay|transfer|message|jama|bhugtan|darj|update)\s+(?:karo|kare|karen|karein|kariye|karna)|(?:jawab|jawaab)\s+do"),
    "credential": phrase(r"ओटीपी|ओ टी पी|पासवर्ड|पिन|सीवीवी|गुप्त कोड|सत्यापन कोड|ओटिपी|पासवर्ड्स|gupt code|gupt kod"),
    "disclose": phrase(r"भेजें|भेजो|भेजिए|भेजिये|बताएं|बताएँ|बताओ|बताइए|बताइये|दें|देना|(?:साझा|शेयर|दर्ज)\s*(?:करें|करो|कीजिए)|bhejo|bheje|bhejen|bhejein|bhejiye|batao|bataye|batayen|batayein|bataiye|(?:share|saajha|sajha|sms|darj)\s+(?:karo|kare|karen|karein|kariye|karna)"),
    "fee": phrase(r"प्रोसेसिंग फीस|पंजीकरण शुल्क|रजिस्ट्रेशन फीस|अग्रिम शुल्क|शुल्क|फीस|शुल्क का भुगतान|registration fees?|processing fees?|shulk|fees|pahle paise|pehle paise"),
    "authority": phrase(r"प्रधानमंत्री|प्रधान मंत्री|सरकार|सरकारी|रिजर्व बैंक|रिज़र्व बैंक|अधिकारी|pradhanmantri|pradhan mantri|sarkar|sarkaar|sarkari|sarkaari|adhikari|govt"),
    "prize": phrase(r"लॉटरी|लाटरी|इनाम|पुरस्कार|विजेता|जीत गए|जीता|जीती|jeet gaye|jeeta|jeeti|jeete|inaam|inam|vijeta"),
    "warning": phrase(r"सावधान|फर्जी (?:संदेश|मैसेज|एसएमएस)|फ़र्ज़ी (?:संदेश|मैसेज)|धोखाधड़ी का उदाहरण|ठगी का उदाहरण|स्कैम का उदाहरण|फिशिंग का उदाहरण|saavdhan|savdhan|farzi (?:message|sms|msg)|farji (?:message|sms|msg)|fraud ka example|scam ka example|dhokhadhadi ka udaharan"),
}
CONTACT_EXTRA = phrase(r"(?:verify|confirm)\s+(?:karo|kare|karen|karein|kariye|kijiye)|bharo|bharein|bharen|bhariye|jayein|jayen|jaaiye|jaiye|jao|सत्यापन\s*करें|पुष्टि\s*करें")
PATTERNS["identity"] = phrase(r"bank\s+(?:account|details)|account|kyc|pan(?:\s+card)?|aadhaar|aadhar|बैंक\s*(?:खाता|खाते|जानकारी)|खाता|खाते|आधार|पैन|केवाईसी")
PATTERNS["verification"] = phrase(r"verify|verification|confirm|update|reactivat(?:e|ion)|सत्यापन|पुष्टि|अपडेट|सक्रिय|जानकारी\s*भरें")
PATTERNS["coercion"] = phrase(r"block(?:ed)?|suspend(?:ed)?|freez(?:e|en)|frozen|disabl(?:e|ed)|disconnect(?:ed)?|reject(?:ed)?|बंद|निलंबित|ब्लॉक|सस्पेंड|अस्वीकार|रद्द")
CONCEPT_INFO = {**SIGNALS,
    "identity": ("Identity or account context", "The request refers to an account, KYC or an identity document."),
    "verification": ("Verification or update requested", "The text asks for verification, confirmation or an update."),
    "coercion": ("Account or service threat", "Blocking, suspension or rejection is invoked near a verification request; the claim is unverified.")}
# A bare host is evidence of a possible destination, not a verified URL or IOC.
BARE_DESTINATION = re.compile(
    r"(?<![A-Za-z0-9@._-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,24}(?:/[^\s<>]*)?(?![A-Za-z0-9_-])", re.I)
RULES = (
    ("indic_money_response_lure", ("reward", "money", "bank", "contact")),
    ("indic_authority_money_request", ("authority", "reward", "money", "contact")),
    ("indic_prize_response_lure", ("prize", "money", "contact")),
    ("indic_credential_disclosure", ("credential", "disclose", "contact")),
    ("indic_credential_link_request", ("credential", "contact", "link")),
    ("indic_advance_fee_lure", ("reward", "money", "fee", "contact")),
    ("indic_identity_threat_destination", ("identity", "coercion", "verification", "contact", "link")),
)
NEGATION = phrase(r"नहीं(?!\s*तो)|नही(?!\s*तो)|मत|ना|न|nahi(?!\s+toh?\b)|nahin(?!\s+toh?\b)|mat|na|not|never|don't|do not")
CONTRAST = phrase(r"लेकिन|मगर|lekin|magar|but|however")
COMPLETED = phrase(r"जमा हो (?:गए|गये|गया|गई)|क्रेडिट हो (?:गए|गये|गया|गई)|भुगतान सफल|जमा किए गए|जमा किये गये|credited|deposited|paid|jama ho gaya|jama ho gaye|credit ho gaya|credit ho gaya|payment successful")
FUTURE = phrase(r"मिलेंगे|मिलेगा|मिलेगी|पाने|पाएं|पाएँ|प्राप्त करें|दिए जाएंगे|दिये जायेंगे|milega|milenge|milegi|paane|pane|claim|collect|redeem|unlock|giving|receive|दे रही|देगी|दे रहे|de rahi|de rahe|diye jayenge")
ENTER = phrase(r"दर्ज करें|दर्ज करो|भरें|भरो|darj karo|darj karein|darj karen|enter|submit|confirm")


def _clause(text, start, end, radius=65):
    left = re.split(r"[.!?;\n।]", text[max(0, start-radius):start])[-1]
    right = re.split(r"[.!?;\n।]", text[end:end+radius])[0]
    return left, right


def _negated(text, match):
    left, right = _clause(text, match.start(), match.end(), 45)
    # A contrast starts a new instruction; an earlier warning must not erase it.
    fences = list(CONTRAST.finditer(left))
    if fences:
        left = left[fences[-1].end():]
    fence = CONTRAST.search(right)
    if fence:
        right = right[:fence.start()]
    allowed = re.compile(r"(?:\s|कभी|भी|किसी|को|अपना|अपने|अपनी|ओटीपी|पासवर्ड|पिन|otp|password|pin|cvv|bhi|kabhi|kisi|ko|apna|apne|apni)*", re.I)
    before = list(NEGATION.finditer(left))
    prefix_negative = bool(before and allowed.fullmatch(left[before[-1].end():]))
    # A later "no fee" clause must not erase an earlier disclosure request.
    suffix = NEGATION.match(right.lstrip())
    suffix_negative = bool(suffix)
    if suffix and suffix.group().casefold() == "na":
        # "bhejo na" is a polite request; "share na karo" is a prohibition.
        suffix_negative = bool(re.match(r"\s*(?:karo|karein|karen|karna)\b", right.lstrip()[suffix.end():], re.I))
    return prefix_negative or suffix_negative


def _requested(text, match):
    if PATTERNS["contact"].fullmatch(match.group()) or CONTACT_EXTRA.fullmatch(match.group()):
        return True
    left, right = _clause(text, match.start(), match.end())
    # English action + Hindi inflection, e.g. "OTP share mat karo" is negated
    # separately; noun-only mentions of SMS are not requests.
    return bool(_is_request(text, match) or re.match(r"\s*(?:करें|करो|karo|karein|karen|kariye)\b", right, re.I)
                or re.search(r"(?:कृपया|kripya|kripaya)\s*$", left, re.I))


def assess_indic_message(text):
    matches = {}
    for kind in CONCEPT_INFO:
        found = {}
        extra = BARE_DESTINATION if kind == "link" else CONTACT_EXTRA if kind == "contact" else None
        for pattern in (ENGLISH.get(kind), PATTERNS.get(kind), extra):
            if pattern:
                for m in pattern.finditer(text):
                    found[(m.start(), m.end())] = m
        values = sorted(found.values(), key=lambda m: (m.start(), -m.end()))
        if kind == "link":
            # Do not display the same destination again as the host inside a URL.
            outer = []
            for m in values:
                if not outer or m.start() >= outer[-1].end():
                    outer.append(m)
            values = outer
        matches[kind] = values
    for kind in ("reward", "prize", "contact", "disclose", "coercion"):
        matches[kind] = [m for m in matches[kind] if not _negated(text, m)]
    matches["contact"] = [m for m in matches["contact"] if _requested(text, m)]
    positions = {kind: [m.start() for m in values] for kind, values in matches.items()}
    signals, rules, contexts = {}, set(), []
    for anchor in matches["contact"]:
        start, end = max(0, anchor.start()-300), anchor.end()+300
        nearby = {k: [m for m in values[bisect_left(positions[k], start):bisect_left(positions[k], end)] if m.end() <= end]
                  for k, values in matches.items()}
        window = text[start:end]
        for name, required in RULES:
            if not all(nearby[k] for k in required):
                continue
            if name in ("indic_money_response_lure", "indic_authority_money_request", "indic_prize_response_lure"):
                if COMPLETED.search(window) and not FUTURE.search(window):
                    continue
            if name == "indic_credential_disclosure" and not any(not ENTER.fullmatch(m.group()) for m in nearby["disclose"]):
                continue
            previous = [m.end() for m in CONTRAST.finditer(text[start:anchor.start()])]
            context_start = start + previous[-1] if previous else start
            contexts.append(any(context_start <= m.start() <= anchor.start() for m in nearby["warning"]))
            rules.add(name)
            for kind in dict.fromkeys((*required, "phone", "link", "authority", "warning")):
                for m in nearby[kind][:3]:
                    info = CONCEPT_INFO[kind]
                    if kind == "link" and not re.match(r"(?:https?://|hxxps?://|www\.)", m.group(), re.I):
                        info = ("Domain-like destination", "A domain-like address appears near the request. Its syntax and reputation are unverified; it is not a confirmed URL.")
                    signals[(kind, m.start(), m.end())] = {
                        "kind": kind, "label": info[0], "explanation": info[1],
                        "original_start": m.start(), "original_end": m.end(), "original_text": m.group()}
    status = "no_pattern_detected" if not rules else "review_context" if all(contexts) else "suspicious"
    summary = {
        "no_pattern_detected": "No supported Hindi/Hinglish pattern matched. This does not establish safety; trained Hindi/Hinglish models are not available.",
        "review_context": "Hindi/Hinglish scam cues appear in warning or example context. Review the original text.",
        "suspicious": "Possible social engineering: Hindi/Hinglish request and lure patterns detected. Verify the sender independently.",
    }[status]
    return {"rule_version": 2, "indic_rule_version": INDIC_RULE_VERSION, "method": "hindi_hinglish_lexical_rules",
            "status": status, "summary": summary, "matched_rules": sorted(rules),
            "signals": sorted(signals.values(), key=lambda s: (s["original_start"], s["kind"])),
            "limitations": "Authored Hindi/Hinglish/English phrase rules only. No Hindi/Hinglish trained classifier or CTI NER, fact checking, reputation lookup or calibrated probability. Dialects, spelling and context can be missed.",
            "offset_unit": "unicode_code_points", "offset_interval": "half_open"}


def assess_indic_document(doc):
    result = assess_indic_message(doc.text)
    for signal in result["signals"]:
        start, end = doc.original_span(signal["original_start"], signal["original_end"])
        signal.update(original_start=start, original_end=end, original_text=doc.original_text[start:end])
    return result
