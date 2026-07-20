"""
AlphaCare Python AI Service
FastAPI + scikit-learn NLP symptom checker + disease predictor
Enhanced with:
  - Emergency Score (0-100%) calculation
  - Multilingual NLP (Urdu, Punjabi, English auto-detection)
  - First-aid instructions per language
  - Emergency popup message per language
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from ml_model import predictor
from disease_data import DISEASE_DATA, SPEC_LABELS
from symptom_matcher import normalize_symptoms, classify_small_talk
from language_support import (
    detect_language,
    translate_to_english,
    get_first_aid,
    get_emergency_popup_message,
    get_template,
)
import os

app = FastAPI(
    title="AlphaCare AI Service",
    description="NLP-based symptom checker, disease predictor, emergency detector & multilingual support",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Request / Response models 

class SymptomsRequest(BaseModel):
    symptoms: str
    top_n: Optional[int] = 3

class FirstAidStep(BaseModel):
    step: str

class AnalysisResponse(BaseModel):
    predictions: list
    severity: str
    isEmergency: bool
    emergencyScore: float          # 0.0 – 100.0
    emergencyMessage: str
    emergencyPopupMessage: str     # Localized popup text
    firstAidInstructions: list     # Localized first-aid steps
    recommendedSpecKey: str
    recommendedSpecLabel: str
    detectedLanguage: str          # 'en', 'ur', 'pa', 'ar'
    analysisIntro: str             # Localized intro sentence for chat


#  Emergency Score Calculation 

EMERGENCY_SCORE_WEIGHTS = {
    # Definite emergencies (high weight)
    "crushing chest pain": 90,
    "heart attack": 95,
    "stroke": 95,
    "cannot breathe": 90,
    "can't breathe": 90,
    "severe breathing difficulty": 85,
    "loss of consciousness": 90,
    "unconscious": 90,
    "face drooping": 85,
    "arm weakness": 70,
    "slurred speech": 80,
    "paralysis": 88,
    "seizure": 85,
    "accident":71,
    "road accident":71,
    "fall from building":70,
    "convulsion": 85,
    "suicidal thoughts": 80,
    # Urdu equivalents
    "سانس نہیں": 90,
    "بے ہوشی": 90,
    "دورہ": 85,
    "حادثہ":71,
    "منہ ٹیڑھا": 85,
    "ہاتھ کمزور": 70,
    "فالج": 95,
    "دل کا دورہ": 95,
    # Roman Urdu
    "sans nahi": 90,
    "mera accident ho gya ha":71,
    "electric shock":60,
    "high voltage electric shock":75,
    "suicid":50,
    "behoshi": 90,
    "dora": 85,
    "dil ka dora": 95,
    "emergency": 80,
    "madad": 40,
    "dil ka dora": 95,
    "seene me dard": 50,
    "sans nahi": 90,
    "sans nahi aa rahi": 90,
    "behosh": 90,
    "falij": 95,
    "dora": 85,
    "bohat khoon": 95,
    "sar pe chot": 90,
    "accident ho gaya": 70,
    "jal gaya": 70,
    "zehar": 90,
    "saanp ne kata": 85,
    "bijli ka jhatka": 60,

    # High-severity combinations
    "severe chest pain": 90,
    "chest pain": 50,
    "shortness of breath": 35,
    "left arm pain": 55,
    "jaw pain": 50,
    "sweating": 20,
    "rapid heartbeat": 25,
    "palpitations": 18,
    "high fever": 20,
    "severe headache": 20,
    "sudden numbness": 45,
    "confusion": 20,
    "blurred vision": 15,
    "dizziness": 10,
    "blood vomiting": 70,
    "blood vomit": 70,
    "bleeding": 30,
    "severe bleeding": 91,
    "head injury": 93,
    "broken bone": 55,
    "fracture": 55,
    "burn": 40,
    "severe burn": 75,
    "poisoning": 75,
    "overdose": 80,
    "anaphylaxis": 90,
    "allergic reaction": 40,

    # --- New: emergency signals from expanded disease set ---
    "stiff neck": 82,
    "neck stiffness": 78,
    "meningitis": 85,
    "deep vein thrombosis": 78,
    "appendicitis": 78,
    "severe abdominal pain": 65,
    "coughing blood": 78,
    "difficulty breathing": 55,

    # Urdu (script)
    "گردن اکڑنا": 80,
    "خون کی کھانسی": 78,
    "شدید پیٹ درد": 65,
    "خودکشی": 80,

    # Roman Urdu
    "gardan akarna": 80,
    "gardan sakht": 78,
    "khoon ki khansi": 78,
    "shadeed pait dard": 65,
    "khudkushi": 80,
    "khud kushi": 80,
    "sans lene mein mushkil": 55,
    "khoon ka thakka": 70,

    # Roman Punjabi
    "gardan akri hoi": 78,
    "khoon wali khansi": 75,
    "tez pet dard": 60,
    "khoon da thakka": 68,
    "saah lain vich mushkil": 55,
}

# which score is considered an emergency
EMERGENCY_THRESHOLD = 70.0


def calculate_emergency_score(text: str, translated_text: str) -> float:
    """
    Calculate emergency score (0-100) from symptom text.
    Combines keyword weights with symptom co-occurrence boosting.
    """
    combined = (text + " " + translated_text).lower()
    
    score = 0.0
    matched_weights = []

    for keyword, weight in EMERGENCY_SCORE_WEIGHTS.items():
        if keyword in combined:
            matched_weights.append(weight)

    if not matched_weights:
        return 0.0

    # Primary score = highest single match
    primary = max(matched_weights)
    score = primary

    # Co-occurrence boost: multiple emergency signals compound
    if len(matched_weights) >= 2:
        secondary_scores = sorted(matched_weights, reverse=True)[1:]
        # Each additional signal adds diminishing boost (max total +20)
        boost = min(sum(s * 0.15 for s in secondary_scores), 20.0)
        score = min(score + boost, 100.0)

    return round(score, 1)


#  Endpoints 

@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "AlphaCare AI Service",
        "version": "2.0.0",
        "features": ["emergency_score", "multilingual_nlp", "first_aid", "auto_language_detect"],
    }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: SymptomsRequest):
    """
    Accept free-text symptoms in English/Urdu/Punjabi (any phrasing, typos
    included), return disease predictions, emergency classification,
    localized first-aid, and recommended specialist.

    Engine: local TF-IDF + Naive Bayes classifier (as documented in the
    thesis), with a preprocessing layer that normalizes phrasing/typos
    before classification, and a keyword-based emergency safety net.
    """
    raw_text = req.symptoms.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Symptoms text is required")

    result = _analyze_locally(req, raw_text)

    #  Safety net: never let a known emergency phrase slip through 
    local_emergency_score = calculate_emergency_score(raw_text, raw_text)
    if local_emergency_score >= EMERGENCY_THRESHOLD and not result["isEmergency"]:
        result["isEmergency"] = True
        result["severity"] = "emergency"
        result["emergencyScore"] = max(result["emergencyScore"], local_emergency_score)
        result["emergencyMessage"] = get_emergency_popup_message(result["detectedLanguage"])
        result["emergencyPopupMessage"] = get_emergency_popup_message(result["detectedLanguage"])
        if not result["firstAidInstructions"]:
            result["firstAidInstructions"] = get_first_aid(raw_text, result["detectedLanguage"])

    return result


def _analyze_locally(req: "SymptomsRequest", raw_text: str) -> dict:
    """Local TF-IDF + Naive Bayes classifier pipeline - matches the
    architecture documented in Appendix A.1 of the thesis."""
    lang = detect_language(raw_text)
    english_text = translate_to_english(raw_text, lang)
    normalized_text, matched_symptoms = normalize_symptoms(english_text)

    if not matched_symptoms:
        category = classify_small_talk(raw_text)
        if category:
            reply = get_template(f"small_talk_{category}", lang)
        else:
            reply = get_template("no_symptoms", lang)
        return {
            "predictions": [],
            "severity": "",
            "isEmergency": False,
            "emergencyScore": 0.0,
            "emergencyMessage": "",
            "emergencyPopupMessage": "",
            "firstAidInstructions": [],
            "recommendedSpecKey": "general",
            "recommendedSpecLabel": SPEC_LABELS.get("general", "General Physician"),
            "detectedLanguage": lang,
            "analysisIntro": reply,
        }

    predictions = predictor.predict(normalized_text, top_n=req.top_n or 3)
    emergency_score = calculate_emergency_score(raw_text, english_text)
    keyword_emergency = predictor.detect_emergency(english_text) or predictor.detect_emergency(raw_text)

    # A confidently-predicted disease that is inherently emergency-level
    # (per its own metadata, e.g. Meningitis/Appendicitis/Heart Attack)
    # should also count, even if the exact wording didn't match a
    # hardcoded emergency phrase or clear the numeric score threshold.
    top_disease_is_emergency = bool(
        predictions
        and predictor.meta.get(predictions[0]["name"], {}).get("severity") == "emergency"
    )

    is_emergency = (emergency_score >= EMERGENCY_THRESHOLD) or keyword_emergency or top_disease_is_emergency
    if (keyword_emergency or top_disease_is_emergency) and emergency_score < EMERGENCY_THRESHOLD:
        emergency_score = EMERGENCY_THRESHOLD

    severity = predictor.get_severity(predictions, is_emergency)
    rec_spec = predictor.recommended_spec(predictions)

    if is_emergency:
        emergency_msg = get_emergency_popup_message(lang)
        first_aid = get_first_aid(raw_text + " " + english_text, lang)
        analysis_intro = get_template("emergency_intro", lang)
    else:
        emergency_msg = ""
        first_aid = []
        analysis_intro = get_template("analysis_intro", lang)

    return {
        "predictions": predictions,
        "severity": severity,
        "isEmergency": is_emergency,
        "emergencyScore": emergency_score,
        "emergencyMessage": emergency_msg,
        "emergencyPopupMessage": get_emergency_popup_message(lang) if is_emergency else "",
        "firstAidInstructions": first_aid,
        "recommendedSpecKey": rec_spec,
        "recommendedSpecLabel": SPEC_LABELS.get(rec_spec, "General Physician"),
        "detectedLanguage": lang,
        "analysisIntro": analysis_intro,
    }


@app.get("/symptoms")
def get_symptoms():
    """Return all known symptom keywords and disease names for autocomplete."""
    all_symptoms = sorted(list({s for d in DISEASE_DATA for s in d["symptoms"]}))
    diseases = [{"name": d["disease"], "specKey": d["specKey"]} for d in DISEASE_DATA]
    return {"symptoms": all_symptoms, "diseases": diseases}


@app.get("/specialties")
def get_specialties():
    return {"specialties": SPEC_LABELS}


@app.get("/languages")
def get_languages():
    """Return supported languages."""
    return {
        "supported": [
            {"code": "en", "name": "English"},
            {"code": "ur", "name": "Urdu / Roman Urdu"},
            {"code": "pa", "name": "Punjabi"},
            {"code": "ar", "name": "Arabic"},
        ]
    }

















# """
# AlphaCare Python AI Service
# FastAPI + scikit-learn NLP symptom checker + disease predictor
# Enhanced with:
#   - Emergency Score (0-100%) calculation
#   - Multilingual NLP (Urdu, Punjabi, English auto-detection)
#   - First-aid instructions per language
#   - Emergency popup message per language
# Run: uvicorn main:app --reload --port 8000
# """

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional, List
# from ml_model import predictor
# from disease_data import DISEASE_DATA, SPEC_LABELS
# from language_support import (
#     detect_language,
#     translate_to_english,
#     get_first_aid,
#     get_emergency_popup_message,
#     get_template,
# )
# import os

# app = FastAPI(
#     title="AlphaCare AI Service",
#     description="NLP-based symptom checker, disease predictor, emergency detector & multilingual support",
#     version="2.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# #  Request / Response models 

# class SymptomsRequest(BaseModel):
#     symptoms: str
#     top_n: Optional[int] = 3

# class FirstAidStep(BaseModel):
#     step: str

# class AnalysisResponse(BaseModel):
#     predictions: list
#     severity: str
#     isEmergency: bool
#     emergencyScore: float          # 0.0 – 100.0
#     emergencyMessage: str
#     emergencyPopupMessage: str     # Localized popup text
#     firstAidInstructions: list     # Localized first-aid steps
#     recommendedSpecKey: str
#     recommendedSpecLabel: str
#     detectedLanguage: str          # 'en', 'ur', 'pa', 'ar'
#     analysisIntro: str             # Localized intro sentence for chat


# #  Emergency Score Calculation 

# EMERGENCY_SCORE_WEIGHTS = {
#     # Definite emergencies (high weight)
#     "crushing chest pain": 90,
#     "heart attack": 95,
#     "stroke": 95,
#     "cannot breathe": 90,
#     "can't breathe": 90,
#     "severe breathing difficulty": 85,
#     "loss of consciousness": 90,
#     "unconscious": 90,
#     "face drooping": 85,
#     "arm weakness": 70,
#     "slurred speech": 80,
#     "paralysis": 88,
#     "seizure": 85,
#     "accident":71,
#     "road accident":71,
#     "fall from building":70,
#     "convulsion": 85,
#     "suicidal thoughts": 80,
#     # Urdu equivalents
#     "سانس نہیں": 90,
#     "بے ہوشی": 90,
#     "دورہ": 85,
#     "حادثہ":71,
#     "منہ ٹیڑھا": 85,
#     "ہاتھ کمزور": 70,
#     "فالج": 95,
#     "دل کا دورہ": 95,
#     # Roman Urdu
#     "sans nahi": 90,
#     "mera accident ho gya ha":71,
#     "electric shock":60,
#     "high voltage electric shock":75,
#     "suicid":50,
#     "behoshi": 90,
#     "dora": 85,
#     "dil ka dora": 95,
#     "emergency": 80,
#     "madad": 40,
#     "dil ka dora": 95,
#     "seene me dard": 50,
#     "sans nahi": 90,
#     "sans nahi aa rahi": 90,
#     "behosh": 90,
#     "falij": 95,
#     "dora": 85,
#     "bohat khoon": 95,
#     "sar pe chot": 90,
#     "accident ho gaya": 70,
#     "jal gaya": 70,
#     "zehar": 90,
#     "saanp ne kata": 85,
#     "bijli ka jhatka": 60,

#     # High-severity combinations
#     "severe chest pain": 90,
#     "chest pain": 50,
#     "shortness of breath": 35,
#     "left arm pain": 55,
#     "jaw pain": 50,
#     "sweating": 20,
#     "rapid heartbeat": 25,
#     "palpitations": 18,
#     "high fever": 20,
#     "severe headache": 20,
#     "sudden numbness": 45,
#     "confusion": 20,
#     "blurred vision": 15,
#     "dizziness": 10,
#     "blood vomiting": 70,
#     "blood vomit": 70,
#     "bleeding": 30,
#     "severe bleeding": 91,
#     "head injury": 93,
#     "broken bone": 55,
#     "fracture": 55,
#     "burn": 40,
#     "severe burn": 75,
#     "poisoning": 75,
#     "overdose": 80,
#     "anaphylaxis": 90,
#     "allergic reaction": 40,
# }

# # which score is considered an emergency
# EMERGENCY_THRESHOLD = 70.0


# def calculate_emergency_score(text: str, translated_text: str) -> float:
#     """
#     Calculate emergency score (0-100) from symptom text.
#     Combines keyword weights with symptom co-occurrence boosting.
#     """
#     combined = (text + " " + translated_text).lower()
    
#     score = 0.0
#     matched_weights = []

#     for keyword, weight in EMERGENCY_SCORE_WEIGHTS.items():
#         if keyword in combined:
#             matched_weights.append(weight)

#     if not matched_weights:
#         return 0.0

#     # Primary score = highest single match
#     primary = max(matched_weights)
#     score = primary

#     # Co-occurrence boost: multiple emergency signals compound
#     if len(matched_weights) >= 2:
#         secondary_scores = sorted(matched_weights, reverse=True)[1:]
#         # Each additional signal adds diminishing boost (max total +20)
#         boost = min(sum(s * 0.15 for s in secondary_scores), 20.0)
#         score = min(score + boost, 100.0)

#     return round(score, 1)


# #  Endpoints 

# @app.get("/health")
# def health():
#     return {
#         "status": "online",
#         "service": "AlphaCare AI Service",
#         "version": "2.0.0",
#         "features": ["emergency_score", "multilingual_nlp", "first_aid", "auto_language_detect"],
#     }


# @app.post("/analyze", response_model=AnalysisResponse)
# def analyze(req: SymptomsRequest):
#     """
#     Accept free-text symptoms (any language), return:
#       - Disease predictions
#       - Emergency score (0-100%)
#       - Emergency classification (score >= 70 = emergency)
#       - Localized first-aid instructions
#       - Localized emergency popup message
#       - Detected language
#     """
#     raw_text = req.symptoms.strip()
#     if not raw_text:
#         raise HTTPException(status_code=400, detail="Symptoms text is required")

#     #  Step 1: Language Detection 
#     lang = detect_language(raw_text)

#     #  Step 2: Translate to English for ML model 
#     english_text = translate_to_english(raw_text, lang)

#     #  Step 3: ML Predictions (use English text) 
#     predictions = predictor.predict(english_text, top_n=req.top_n or 3)

    

#     #  Step 4: Emergency Score 
#     emergency_score = calculate_emergency_score(raw_text, english_text)

#     #  Step 5: Emergency Classification (score-based) 
#     # Also honour the boolean keyword detector for backwards compat
#     keyword_emergency = predictor.detect_emergency(english_text) or predictor.detect_emergency(raw_text)
#     is_emergency = (emergency_score >= EMERGENCY_THRESHOLD) or keyword_emergency

#     # If keyword triggered but score was low, set minimum score
#     if keyword_emergency and emergency_score < EMERGENCY_THRESHOLD:
#         emergency_score = EMERGENCY_THRESHOLD

#     #  Step 6: Severity 
#     severity = predictor.get_severity(predictions, is_emergency)

#     #  Step 7: Recommended Specialist 
#     rec_spec = predictor.recommended_spec(predictions)

#     #  Step 8: Localized Messages 
#     if is_emergency:
#         emergency_msg = get_emergency_popup_message(lang)
#         first_aid = get_first_aid(raw_text + " " + english_text, lang)
#         analysis_intro = get_template("emergency_intro", lang)
#     else:
#         emergency_msg = ""
#         first_aid = []
#         analysis_intro = get_template("analysis_intro", lang)

#     return {
#         "predictions":          predictions,
#         "severity":             severity,
#         "isEmergency":          is_emergency,
#         "emergencyScore":       emergency_score,
#         "emergencyMessage":     emergency_msg,
#         "emergencyPopupMessage": get_emergency_popup_message(lang) if is_emergency else "",
#         "firstAidInstructions": first_aid,
#         "recommendedSpecKey":   rec_spec,
#         "recommendedSpecLabel": SPEC_LABELS.get(rec_spec, "General Physician"),
#         "detectedLanguage":     lang,
#         "analysisIntro":        analysis_intro,
#     }


# @app.get("/symptoms")
# def get_symptoms():
#     """Return all known symptom keywords and disease names for autocomplete."""
#     all_symptoms = sorted(list({s for d in DISEASE_DATA for s in d["symptoms"]}))
#     diseases = [{"name": d["disease"], "specKey": d["specKey"]} for d in DISEASE_DATA]
#     return {"symptoms": all_symptoms, "diseases": diseases}


# @app.get("/specialties")
# def get_specialties():
#     return {"specialties": SPEC_LABELS}


# @app.get("/languages")
# def get_languages():
#     """Return supported languages."""
#     return {
#         "supported": [
#             {"code": "en", "name": "English"},
#             {"code": "ur", "name": "Urdu / Roman Urdu"},
#             {"code": "pa", "name": "Punjabi"},
#             {"code": "ar", "name": "Arabic"},
#         ]
#     }















