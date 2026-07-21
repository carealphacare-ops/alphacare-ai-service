import re
import numpy as np
from rapidfuzz import process 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from disease_data import DISEASE_DATA, EMERGENCY_KEYWORDS

#  Preprocess 

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

#  Training Data 

def build_training_data():
    X, y = [], []
    for entry in DISEASE_DATA:
        base = " ".join(entry["symptoms"])
        X.append(base)
        y.append(entry["disease"])

        for s in entry["symptoms"]:
            X.append(s)
            y.append(entry["disease"])

    return X, y

def build_symptom_list():
    symptoms = []

    for disease in DISEASE_DATA:
        symptoms.extend(disease["symptoms"])

    # duplicate remove
    return list(set(symptoms))

#  Model 

class DiseasePredictor:
    def __init__(self):
        X, y = build_training_data()
        X = [preprocess(x) for x in X]

        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
            ("clf", MultinomialNB())
        ])

        self.pipeline.fit(X, y_enc)

        self.meta = {d["disease"]: d for d in DISEASE_DATA}
        self.all_symptoms = build_symptom_list()

    #  Prediction 

    def predict(self, text, top_n=3):
        text = preprocess(text)
        probs = self.pipeline.predict_proba([text])[0]
        idxs = np.argsort(probs)[::-1][:top_n]

        results = []
        for i in idxs:
            disease = self.le.inverse_transform([i])[0]
            confidence = round(float(probs[i]) * 100, 1)

            meta = self.meta.get(disease, {})

            results.append({
                "name": disease,
                "confidence": min(confidence * 2.5, 95),
                "description": meta.get("description", ""),
                "specKey": meta.get("specKey", "general")
            })

        return results

    #  FIXED EMERGENCY LOGIC 

    def detect_emergency(self, text: str) -> bool:
        text = text.lower()

        # strict matching only
        for kw in EMERGENCY_KEYWORDS:
            if kw in text:
                return True

        return False

    #  Severity 

    def get_severity(self, predictions, is_emergency):
        if is_emergency:
            return "emergency"

        if not predictions:
            return "mild"

        top = predictions[0]["name"]
        raw_severity = self.meta.get(top, {}).get("severity", "mild")

        # Never show an "emergency" severity badge unless the emergency flag
        # actually fired - showing that label without the alert/first-aid
        # popup is confusing and looks like a bug to the patient.
        if raw_severity == "emergency":
            return "moderate"
        return raw_severity

    #  Recommended Specialist 

    def recommended_spec(self, predictions):
        if not predictions:
            return "general"

        # Confidence-weighted vote across all returned predictions, not just
        # the top one - if #2 and #3 both point to the same specialist and
        # collectively outweigh a narrowly-ahead #1, that's a more reliable
        # signal than blindly trusting whichever prediction happened to win
        # the classifier's argmax by a small margin.
        spec_scores = {}
        for p in predictions:
            spec_scores[p["specKey"]] = spec_scores.get(p["specKey"], 0) + p["confidence"]

        return max(spec_scores, key=spec_scores.get)


# Singleton
predictor = DiseasePredictor()








# import re
# import numpy as np
# from rapidfuzz import process 
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder
# from disease_data import DISEASE_DATA, EMERGENCY_KEYWORDS

# #  Preprocess 

# def preprocess(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     return text

# #  Training Data 

# def build_training_data():
#     X, y = [], []
#     for entry in DISEASE_DATA:
#         base = " ".join(entry["symptoms"])
#         X.append(base)
#         y.append(entry["disease"])

#         for s in entry["symptoms"]:
#             X.append(s)
#             y.append(entry["disease"])

#     return X, y

# def build_symptom_list():
#     symptoms = []

#     for disease in DISEASE_DATA:
#         symptoms.extend(disease["symptoms"])

#     # duplicate remove
#     return list(set(symptoms))

# #  Model 

# class DiseasePredictor:
#     def __init__(self):
#         X, y = build_training_data()
#         X = [preprocess(x) for x in X]

#         self.le = LabelEncoder()
#         y_enc = self.le.fit_transform(y)

#         self.pipeline = Pipeline([
#             ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
#             ("clf", MultinomialNB())
#         ])

#         self.pipeline.fit(X, y_enc)

#         self.meta = {d["disease"]: d for d in DISEASE_DATA}
#         self.all_symptoms = build_symptom_list()

#     #  Prediction 

#     def predict(self, text, top_n=3):
#         text = preprocess(text)
#         probs = self.pipeline.predict_proba([text])[0]
#         idxs = np.argsort(probs)[::-1][:top_n]

#         results = []
#         for i in idxs:
#             disease = self.le.inverse_transform([i])[0]
#             confidence = round(float(probs[i]) * 100, 1)

#             meta = self.meta.get(disease, {})

#             results.append({
#                 "name": disease,
#                 "confidence": min(confidence * 2.5, 95),
#                 "description": meta.get("description", ""),
#                 "specKey": meta.get("specKey", "general")
#             })

#         return results

#     #  FIXED EMERGENCY LOGIC 

#     def detect_emergency(self, text: str) -> bool:
#         text = text.lower()

#         # strict matching only
#         for kw in EMERGENCY_KEYWORDS:
#             if kw in text:
#                 return True

#         return False

#     #  Severity 

#     def get_severity(self, predictions, is_emergency):
#         if is_emergency:
#             return "emergency"

#         if not predictions:
#             return "mild"

#         top = predictions[0]["name"]
#         return self.meta.get(top, {}).get("severity", "mild")

#     #  Recommended Specialist 

#     def recommended_spec(self, predictions):
#         if not predictions:
#             return "general"

#         return predictions[0].get("specKey", "general")


# # Singleton
# predictor = DiseasePredictor()





