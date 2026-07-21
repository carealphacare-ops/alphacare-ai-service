"""
Symptom text normalizer.

Turns messy, free-typed patient input into the clean symptom phrases the
ML model was actually trained on. Handles:
  - different word order    ("pain in heart" / "heart pain" / "heartpain")
  - everyday synonyms       ("tummy ache" -> "abdominal pain")
  - spelling mistakes       ("hedache" -> "headache") via rapidfuzz
  - casual/greeting text    (returns no matches, so the caller can reply
                              conversationally instead of forcing a diagnosis)
"""
import re
from rapidfuzz import fuzz, process
from disease_data import DISEASE_DATA

# ── Canonical vocabulary: the exact symptom phrases the model knows ────────
CANONICAL_SYMPTOMS = sorted({
    s.strip().strip(",") for d in DISEASE_DATA for s in d["symptoms"]
})

# ── Hand-curated everyday synonyms -> canonical symptom ─────────────────────
# Add more lines here any time you notice the bot missing a common phrase.
SYMPTOM_ALIASES = {
    "heart pain": "chest pain", "heart ache": "chest pain", "heartache": "chest pain",
    "my heart hurts": "chest pain", "heart hurts": "chest pain", "heart burning": "chest pain",
    "pain in heart": "chest pain", "pain in my heart": "chest pain", "heartpain": "chest pain",
    "painheart": "chest pain", "heart is paining": "chest pain", "heart paining": "chest pain",
    "tummy ache": "abdominal pain", "tummy pain": "abdominal pain", "belly pain": "abdominal pain",
    "belly ache": "abdominal pain", "stomach ache": "abdominal pain", "stomach pain": "abdominal pain",
    "throwing up": "vomiting", "throw up": "vomiting", "puking": "vomiting",
    "cant sleep": "sleep difficulty", "can't sleep": "sleep difficulty", "trouble sleeping": "sleep difficulty",
    "loose motion": "diarrhea", "loose motions": "diarrhea", "runny tummy": "diarrhea",
    "feeling dizzy": "dizziness", "head spinning": "dizziness",
    "feeling tired": "fatigue", "no energy": "fatigue", "low energy": "fatigue", "exhausted": "fatigue",
    "skin itching": "itching", "itchy skin": "itching",
    "high temperature": "fever", "feeling hot": "fever", "running a temperature": "fever",
    "burning pee": "burning urination", "burning while peeing": "burning urination",
    "peeing a lot": "frequent urination", "peeing frequently": "frequent urination",
    "out of breath": "shortness of breath", "cant breathe": "shortness of breath",
    "cant catch my breath": "shortness of breath", "breathless": "shortness of breath",
    "head hurts": "headache", "head pain": "headache", "head ache": "headache",
    "throat pain": "sore throat", "throat hurts": "sore throat", "sore throat pain": "sore throat",
    "back hurts": "back pain", "joint hurts": "joint pain", "joints hurt": "joint pain",
    "cant move": "inability to move", "skin rash pain": "skin rash",
}


def _reorderings(phrase: str):
    """For 2-word symptoms, also accept 'B A', 'B in A', and the joined 'AB'/'BA' forms."""
    words = phrase.split()
    if len(words) != 2:
        return set()
    a, b = words
    return {f"{b} {a}", f"{b} in {a}", f"{a}{b}", f"{b}{a}"}


# variant phrase -> canonical symptom
_PHRASE_LOOKUP = {}
for s in CANONICAL_SYMPTOMS:
    _PHRASE_LOOKUP[s] = s
    for v in _reorderings(s):
        _PHRASE_LOOKUP.setdefault(v, s)
for alias, canonical in SYMPTOM_ALIASES.items():
    _PHRASE_LOOKUP[alias] = canonical

_ALL_VARIANTS = list(_PHRASE_LOOKUP.keys())

# Split variants by word count so typo-correction never lets a single stray
# word (like "heart") falsely match a longer phrase (like "heart attack").
_SINGLE_WORD_VARIANTS = [v for v in _ALL_VARIANTS if len(v.split()) == 1]
_MULTI_WORD_VARIANTS = [v for v in _ALL_VARIANTS if len(v.split()) > 1]

SMALL_TALK_KEYWORDS = {
    # order matters: more specific categories are checked before generic ones
    "how_are_you": [
        "how are you", "how r u", "hows it going", "how's it going",
        "whats up", "what's up", "kaisay ho", "kaisi ho", "kya hal hai", "kya haal hai",
    ],
    "capability": [
        "what can you do", "who are you", "what are you", "how do you work",
        "how does this work", "kya kar saktay ho", "tum kon ho", "aap kaun ho",
    ],
    "thanks": [
        "thanks", "thank you", "thankyou", "thnx", "shukriya", "meherbani", "mehrbani",
    ],
    "bye": [
        "bye", "goodbye", "good bye", "see you", "tata", "khuda hafiz", "allah hafiz",
        "phir milte hain",
    ],
    "salam": [
        "salam", "saalam", "salaam", "assalam o alaikum", "assalamualaikum",
        "asalam o alikum", "asalamualaikum", "assalamu alaikum", "aoa",
        "salam alaikum",
    ],
    "greeting": [
        "hi", "hii", "hiii", "hello", "hey", "heya", "yo", "namaste",
        "sat sri akal", "good morning", "good evening", "good afternoon",
    ],
    "ok": [
        "ok", "okay", "fine", "alright", "acha", "theek hai", "thik hai",
    ],
}


def classify_small_talk(raw_text: str):
    """
    Detects casual/conversational messages (greetings, thanks, bye, etc.)
    that are not symptom descriptions. Returns a category name (matching a
    'small_talk_<category>' template key) or None if the text doesn't look
    like small talk at all.
    """
    cleaned = raw_text.lower().replace("'", "")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    padded = f" {cleaned} "

    for category, phrases in SMALL_TALK_KEYWORDS.items():
        for phrase in phrases:
            phrase_clean = phrase.replace("'", "")
            if f" {phrase_clean} " in padded or padded.strip() == phrase_clean:
                return category
    return None


def is_greeting(raw_text: str) -> bool:
    """Kept for backward compatibility - use classify_small_talk for richer replies."""
    return classify_small_talk(raw_text) == "greeting"


def normalize_symptoms(text: str, fuzzy_threshold: int = 84, phrase_fuzzy_threshold: int = 78):
    """
    Returns (normalized_text, matched_symptoms).
    normalized_text: matched canonical symptoms joined together — feed this to the classifier.
    matched_symptoms: list of canonical symptoms actually recognized (empty if none found).
    """
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    matched = []
    remaining = f" {cleaned} "

    # 1) exact / synonym / reordered phrase matches (longest phrases first)
    for variant in sorted(_ALL_VARIANTS, key=len, reverse=True):
        token = f" {variant} "
        if token in remaining:
            matched.append(_PHRASE_LOOKUP[variant])
            remaining = remaining.replace(variant, " ")

    # 2) fuzzy match on whatever text is left over (catches spelling mistakes)
    leftover_words = [w for w in remaining.split() if len(w) > 3]

    # 2a) single misspelled word -> single-word symptom (e.g. "hedache" -> "headache")
    # Uses plain ratio (not WRatio) so a short word can't "partial match" a longer phrase.
    used_words = set()
    for word in leftover_words:
        if not _SINGLE_WORD_VARIANTS:
            break
        best = process.extractOne(word, _SINGLE_WORD_VARIANTS, scorer=fuzz.ratio)
        if best and best[1] >= fuzzy_threshold:
            matched.append(_PHRASE_LOOKUP[best[0]])
            used_words.add(word)

    # 2b) misspelled multi-word phrase -> compare 2/3-word windows of the leftover
    # text against known multi-word phrases (e.g. "chets pian" -> "chest pain").
    remaining_words = [w for w in leftover_words if w not in used_words]
    for size in (3, 2):
        i = 0
        while i <= len(remaining_words) - size:
            window = " ".join(remaining_words[i:i + size])
            best = process.extractOne(window, _MULTI_WORD_VARIANTS, scorer=fuzz.ratio)
            if best and best[1] >= phrase_fuzzy_threshold:
                matched.append(_PHRASE_LOOKUP[best[0]])
                del remaining_words[i:i + size]
            else:
                i += 1

    matched = list(dict.fromkeys(matched))  # dedupe, preserve order
    normalized_text = " ".join(matched) if matched else cleaned
    return normalized_text, matched




# """
# Symptom text normalizer.

# Turns messy, free-typed patient input into the clean symptom phrases the
# ML model was actually trained on. Handles:
#   - different word order    ("pain in heart" / "heart pain" / "heartpain")
#   - everyday synonyms       ("tummy ache" -> "abdominal pain")
#   - spelling mistakes       ("hedache" -> "headache") via rapidfuzz
#   - casual/greeting text    (returns no matches, so the caller can reply
#                               conversationally instead of forcing a diagnosis)
# """
# import re
# from rapidfuzz import fuzz, process
# from disease_data import DISEASE_DATA

# # ── Canonical vocabulary: the exact symptom phrases the model knows ────────
# CANONICAL_SYMPTOMS = sorted({
#     s.strip().strip(",") for d in DISEASE_DATA for s in d["symptoms"]
# })

# # ── Hand-curated everyday synonyms -> canonical symptom ─────────────────────
# # Add more lines here any time you notice the bot missing a common phrase.
# SYMPTOM_ALIASES = {
#     "heart pain": "chest pain", "heart ache": "chest pain", "heartache": "chest pain",
#     "my heart hurts": "chest pain", "heart hurts": "chest pain", "heart burning": "chest pain",
#     "pain in heart": "chest pain", "pain in my heart": "chest pain", "heartpain": "chest pain",
#     "painheart": "chest pain", "heart is paining": "chest pain", "heart paining": "chest pain",
#     "tummy ache": "abdominal pain", "tummy pain": "abdominal pain", "belly pain": "abdominal pain",
#     "belly ache": "abdominal pain", "stomach ache": "abdominal pain", "stomach pain": "abdominal pain",
#     "throwing up": "vomiting", "throw up": "vomiting", "puking": "vomiting",
#     "cant sleep": "sleep difficulty", "can't sleep": "sleep difficulty", "trouble sleeping": "sleep difficulty",
#     "loose motion": "diarrhea", "loose motions": "diarrhea", "runny tummy": "diarrhea",
#     "feeling dizzy": "dizziness", "head spinning": "dizziness",
#     "feeling tired": "fatigue", "no energy": "fatigue", "low energy": "fatigue", "exhausted": "fatigue",
#     "skin itching": "itching", "itchy skin": "itching",
#     "high temperature": "fever", "feeling hot": "fever", "running a temperature": "fever",
#     "burning pee": "burning urination", "burning while peeing": "burning urination",
#     "peeing a lot": "frequent urination", "peeing frequently": "frequent urination",
#     "out of breath": "shortness of breath", "cant breathe": "shortness of breath",
#     "cant catch my breath": "shortness of breath", "breathless": "shortness of breath",
#     "head hurts": "headache", "head pain": "headache", "head ache": "headache",
#     "throat pain": "sore throat", "throat hurts": "sore throat", "sore throat pain": "sore throat",
#     "back hurts": "back pain", "joint hurts": "joint pain", "joints hurt": "joint pain",
#     "cant move": "inability to move", "skin rash pain": "skin rash",
# }


# def _reorderings(phrase: str):
#     """For 2-word symptoms, also accept 'B A', 'B in A', and the joined 'AB'/'BA' forms."""
#     words = phrase.split()
#     if len(words) != 2:
#         return set()
#     a, b = words
#     return {f"{b} {a}", f"{b} in {a}", f"{a}{b}", f"{b}{a}"}


# # variant phrase -> canonical symptom
# _PHRASE_LOOKUP = {}
# for s in CANONICAL_SYMPTOMS:
#     _PHRASE_LOOKUP[s] = s
#     for v in _reorderings(s):
#         _PHRASE_LOOKUP.setdefault(v, s)
# for alias, canonical in SYMPTOM_ALIASES.items():
#     _PHRASE_LOOKUP[alias] = canonical

# _ALL_VARIANTS = list(_PHRASE_LOOKUP.keys())

# # Split variants by word count so typo-correction never lets a single stray
# # word (like "heart") falsely match a longer phrase (like "heart attack").
# _SINGLE_WORD_VARIANTS = [v for v in _ALL_VARIANTS if len(v.split()) == 1]
# _MULTI_WORD_VARIANTS = [v for v in _ALL_VARIANTS if len(v.split()) > 1]

# SMALL_TALK_KEYWORDS = {
#     # order matters: more specific categories are checked before generic ones
#     "how_are_you": [
#         "how are you", "how r u", "hows it going", "how's it going",
#         "whats up", "what's up", "kaisay ho", "kaisi ho", "kya hal hai", "kya haal hai",
#     ],
#     "capability": [
#         "what can you do", "who are you", "what are you", "how do you work",
#         "how does this work", "kya kar saktay ho", "tum kon ho", "aap kaun ho",
#     ],
#     "thanks": [
#         "thanks", "thank you", "thankyou", "thnx", "shukriya", "meherbani", "mehrbani",
#     ],
#     "bye": [
#         "bye", "goodbye", "good bye", "see you", "tata", "khuda hafiz", "allah hafiz",
#         "phir milte hain",
#     ],
#     "greeting": [
#         "hi", "hii", "hiii", "hello", "hey", "heya", "yo", "salam", "salaam",
#         "assalam o alaikum", "assalamualaikum", "asalam o alikum", "namaste",
#         "sat sri akal", "good morning", "good evening", "good afternoon",
#     ],
#     "ok": [
#         "ok", "okay", "fine", "alright", "acha", "theek hai", "thik hai",
#     ],
# }


# def classify_small_talk(raw_text: str):
#     """
#     Detects casual/conversational messages (greetings, thanks, bye, etc.)
#     that are not symptom descriptions. Returns a category name (matching a
#     'small_talk_<category>' template key) or None if the text doesn't look
#     like small talk at all.
#     """
#     cleaned = raw_text.lower().replace("'", "")
#     cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
#     cleaned = re.sub(r"\s+", " ", cleaned).strip()
#     padded = f" {cleaned} "

#     for category, phrases in SMALL_TALK_KEYWORDS.items():
#         for phrase in phrases:
#             phrase_clean = phrase.replace("'", "")
#             if f" {phrase_clean} " in padded or padded.strip() == phrase_clean:
#                 return category
#     return None


# def is_greeting(raw_text: str) -> bool:
#     """Kept for backward compatibility - use classify_small_talk for richer replies."""
#     return classify_small_talk(raw_text) == "greeting"


# def normalize_symptoms(text: str, fuzzy_threshold: int = 84, phrase_fuzzy_threshold: int = 78):
#     """
#     Returns (normalized_text, matched_symptoms).
#     normalized_text: matched canonical symptoms joined together — feed this to the classifier.
#     matched_symptoms: list of canonical symptoms actually recognized (empty if none found).
#     """
#     lowered = text.lower().strip()
#     cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
#     cleaned = re.sub(r"\s+", " ", cleaned).strip()

#     matched = []
#     remaining = f" {cleaned} "

#     # 1) exact / synonym / reordered phrase matches (longest phrases first)
#     for variant in sorted(_ALL_VARIANTS, key=len, reverse=True):
#         token = f" {variant} "
#         if token in remaining:
#             matched.append(_PHRASE_LOOKUP[variant])
#             remaining = remaining.replace(variant, " ")

#     # 2) fuzzy match on whatever text is left over (catches spelling mistakes)
#     leftover_words = [w for w in remaining.split() if len(w) > 3]

#     # 2a) single misspelled word -> single-word symptom (e.g. "hedache" -> "headache")
#     # Uses plain ratio (not WRatio) so a short word can't "partial match" a longer phrase.
#     used_words = set()
#     for word in leftover_words:
#         if not _SINGLE_WORD_VARIANTS:
#             break
#         best = process.extractOne(word, _SINGLE_WORD_VARIANTS, scorer=fuzz.ratio)
#         if best and best[1] >= fuzzy_threshold:
#             matched.append(_PHRASE_LOOKUP[best[0]])
#             used_words.add(word)

#     # 2b) misspelled multi-word phrase -> compare 2/3-word windows of the leftover
#     # text against known multi-word phrases (e.g. "chets pian" -> "chest pain").
#     remaining_words = [w for w in leftover_words if w not in used_words]
#     for size in (3, 2):
#         i = 0
#         while i <= len(remaining_words) - size:
#             window = " ".join(remaining_words[i:i + size])
#             best = process.extractOne(window, _MULTI_WORD_VARIANTS, scorer=fuzz.ratio)
#             if best and best[1] >= phrase_fuzzy_threshold:
#                 matched.append(_PHRASE_LOOKUP[best[0]])
#                 del remaining_words[i:i + size]
#             else:
#                 i += 1

#     matched = list(dict.fromkeys(matched))  # dedupe, preserve order
#     normalized_text = " ".join(matched) if matched else cleaned
#     return normalized_text, matched
