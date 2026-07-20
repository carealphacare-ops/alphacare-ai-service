"""
Multilingual NLP support for AlphaCare.
Detects language from user input and provides:
  - Symptom keyword translation to English for model
  - First-aid instructions in detected language
  - Emergency messages in detected language
"""

#  Language Detection Patterns 

# Urdu common words / characters
URDU_PATTERNS = [
    'درد', 'بخار', 'کھانسی', 'سردرد', 'سانس', 'چکر', 'الٹی', 'متلی',
    'دل', 'سینہ', 'پیٹ', 'کمزوری', 'تھکاوٹ', 'بیمار', 'مرض', 'علامت',
    'ہاں', 'نہیں', 'ہے', 'میں', 'مجھے', 'مریض', 'دوائی', 'ڈاکٹر',
    'ہسپتال', 'کیا', 'ہوں', 'بہت', 'ٹھیک', 'تکلیف',
]

# Punjabi common words
PUNJABI_PATTERNS = [
    'ਦਰਦ', 'ਬੁਖਾਰ', 'ਖੰਘ', 'ਸਿਰਦਰਦ', 'ਚੱਕਰ', 'ਉਲਟੀ',
    'ਦਿਲ', 'ਛਾਤੀ', 'ਕਮਜ਼ੋਰੀ', 'ਥਕਾਵਟ', 'ਬਿਮਾਰ',
    # Punjabi in Roman/Shahmukhi
    'dard', 'bukhaar', 'khansi', 'sir dard', 'chakkar', 'ulti',
    'thand', 'garmi', 'beemar', 'takleef', 'dil', 'chaati',
]

# Arabic patterns (bonus)
ARABIC_PATTERNS = [
    'ألم', 'حمى', 'سعال', 'صداع', 'دوار', 'غثيان', 'قلب',
]


#  Urdu Symptom Translations → English 

URDU_SYMPTOM_MAP = {
    # Pain
    'درد': 'pain',
    'سردرد': 'headache',
    'سر درد': 'headache',
    'سینے میں درد': 'chest pain',
    'سینے کا درد': 'chest pain',
    'پیٹ درد': 'abdominal pain',
    'پیٹ میں درد': 'abdominal pain',
    'کمر درد': 'back pain',
    'جوڑوں کا درد': 'joint pain',
    'گلے میں درد': 'sore throat',
    'گلا خراب': 'sore throat',

    # Fever/Temperature
    'بخار': 'fever',
    'تیز بخار': 'high fever',
    'ہلکا بخار': 'mild fever',

    # Respiratory
    'کھانسی': 'cough',
    'سانس لینے میں تکلیف': 'shortness of breath',
    'سانس پھولنا': 'shortness of breath',
    'ناک بہنا': 'runny nose',
    'چھینکیں': 'sneezing',
    'سانس نہیں آ رہی': 'cannot breathe',
    'سانس نہیں آتی': 'cannot breathe',

    # Digestive
    'الٹی': 'vomiting',
    'متلی': 'nausea',
    'دست': 'diarrhea',
    'قبض': 'constipation',
    'بدہضمی': 'indigestion',

    # General
    'چکر': 'dizziness',
    'چکر آنا': 'dizziness',
    'کمزوری': 'weakness',
    'تھکاوٹ': 'fatigue',
    'تھکان': 'fatigue',
    'نیند نہ آنا': 'insomnia',
    'بھوک نہ لگنا': 'loss of appetite',
    'وزن کم ہونا': 'weight loss',
    'جلد پر دانے': 'skin rash',
    'خارش': 'itching',
    'دھڑکن تیز': 'rapid heartbeat',
    'منہ خشک': 'dry mouth',
    'آنکھیں پیلی': 'yellow eyes',
    'یرقان': 'jaundice',
    'پسینہ': 'sweating',
    'کانپنا': 'shivering',
    'دورہ': 'seizure',
    'بے ہوشی': 'loss of consciousness',
    'منہ ٹیڑھا': 'face drooping',
    'ہاتھ کمزور': 'arm weakness',
    'بولنے میں تکلیف': 'slurred speech',
    'خودکشی': 'suicidal thoughts',
    'خودکشی کے خیالات': 'suicidal thoughts',
    'bijli ka jhatka': 'electric shock',

        
    
    # Roman Urdu keywords
    'dard': 'pain',
    'bukhaar': 'fever',
    'bukhar': 'fever',
    'khansi': 'cough',
    'sar dard': 'headache',
    'seene mein dard': 'chest pain',
    'sine mein dard': 'chest pain',
    'ulti': 'vomiting',
    'accident' :'emergency',
    'matli': 'nausea',
    'chakkar': 'dizziness',
    'kamzori': 'weakness',
    'thakawat': 'fatigue',
    'sans lene mein takleef': 'shortness of breath',
    'sans nahi aa rahi': 'cannot breathe',
    'dast': 'diarrhea',
    'khujli': 'itching',
    'paseena': 'sweating',
    'kampna': 'shivering',
    'behoshi': 'loss of consciousness',
    'dora': 'seizure',
    'beemar': 'sick',
    'takleef': 'discomfort',
    'garmi': 'heat',
    'thand': 'cold',
    'joron ka dard': 'joint pain',
    'peeth dard': 'back pain',
    'pet dard': 'abdominal pain',
    'gala kharab': 'sore throat',
    'naak beh rahi': 'runny nose',
    'khoon': 'bleeding',
    'zeher': 'poison',
    'jal gaya': 'burn',
    'chot': 'injury',
    'haddi toot': 'fracture',

    # --- New vocabulary for expanded disease set ---
    # Respiratory / general infection
    'ذائقہ ختم': 'loss of taste',
    'سونگھنے کی حس ختم': 'loss of smell',
    'zaiqa khatam': 'loss of taste',
    'soonghne ki hiss khatam': 'loss of smell',
    'kaanp rahe hain': 'chills',
    'sردی لگنا': 'chills',
    'sardi lagna': 'chills',
    'raat ko paseena': 'night sweats',
    'خون کی کھانسی': 'coughing blood',
    'khoon ki khansi': 'coughing blood',
    'balgham wali khansi': 'cough with phlegm',
    'saans tez chalna': 'rapid breathing',
    'seene mein jakadan': 'chest tightness',

    # Urinary
    'پیشاب میں جلن': 'burning urination',
    'peshab mein jalan': 'burning urination',
    'bar bar peshab': 'frequent urination',
    'peshab mein dard': 'pelvic pain',
    'peshab ka rang gehra': 'dark urine',

    # Sinus/face
    'chehre mein dard': 'facial pain',
    'naak band': 'nasal congestion',
    'sinus ka dabao': 'sinus pressure',

    # Cardio
    'دھڑکن بے ترتیب': 'irregular heartbeat',
    'dhadkan be tarteeb': 'irregular heartbeat',
    'dil ka phadphadana': 'fluttering in chest',
    'behosh hona': 'fainting',
    'tang mein soojan': 'leg swelling',
    'tang thandi': 'cold feet',
    'tang mein aintan': 'leg cramps',

    # Neuro
    'گردن اکڑنا': 'stiff neck',
    'gardan akarna': 'stiff neck',
    'gardan sakht': 'neck stiffness',
    'روشنی سے تکلیف': 'sensitivity to light',
    'roshni se takleef': 'sensitivity to light',
    'hath sunn hona': 'numbness in hands',
    'paon sunn hona': 'numbness in feet',
    'jhunjhunahat': 'tingling',
    'کپکپی': 'tremor',
    'kapkapi': 'tremor',
    'haath kaanpna': 'shaking hands',
    'chehra latakna': 'facial drooping',
    'lar tapakna': 'drooling',

    # Skin
    'khaal safed dhabbay': 'white patches on skin',
    'khaal chilna': 'peeling skin',
    'raat ko zyada khujli': 'itching worse at night',

    # Ortho
    'کندھے کی اکڑن': 'shoulder stiffness',
    'kandhe ki akarn': 'shoulder stiffness',
    'jodon ki soojan': 'joint swelling',
    'subah ki akarn': 'morning stiffness',
    'tang mein jhanjhanahat': 'tingling in leg',

    # Pedia
    'گال کی سوجن': 'swollen cheeks',
    'gaal ki soojan': 'swollen cheeks',
    'jabray mein dard': 'jaw pain',
    'bhonkne wali khansi': 'barking cough',
    'awaz baithna': 'hoarse voice',
    'moonh mein chhalay': 'mouth sores',

    # Gastro
    'seene mein jalan': 'heartburn',
    'کھٹا ذائقہ': 'sour taste in mouth',
    'khata zaiqa': 'sour taste in mouth',
    'nigalne mein mushkil': 'difficulty swallowing',
    'malashay se khoon': 'rectal bleeding',
    'shadeed pait dard': 'severe abdominal pain',
    'peshab ya pakhane mein zor lagana': 'straining',

    # Psych
    'گھبراہٹ کے دورے': 'panic attacks',
    'ghabrahat ke doray': 'panic attacks',
    'kanpakampi': 'trembling',
    'buray khwab': 'nightmares',
    'purani yaadein': 'intrusive memories',

    # Critical safety terms - Roman script (must also be in the translation
    # map itself, not just the emergency-score list, so language detection
    # and translation both catch it)
    'khudkushi': 'suicidal thoughts',
    'khud kushi': 'suicidal thoughts',
    'apni jaan lena': 'suicidal thoughts',

    # Common spelling variant
    'haath kaanpna': 'shaking hands',
    'hath kaanpna': 'shaking hands',
}

# Punjabi Roman → English
PUNJABI_SYMPTOM_MAP = {
    'dard': 'pain',
    'bukhaar': 'fever',
    'khansi': 'cough',
    'sir dard': 'headache',
    'chakkar': 'dizziness',
    'ulti': 'vomiting',
    'kamzori': 'weakness',
    'thakawat': 'fatigue',
    'dil dard': 'chest pain',
    'pet dard': 'abdominal pain',
    'beemar': 'sick',
    'takleef': 'discomfort',

    # --- Expanded Punjabi vocabulary ---
    'sah lain vich mushkil': 'shortness of breath',
    'sah nai aa reha': 'cannot breathe',
    'dast': 'diarrhea',
    'khujli': 'itching',
    'paseena': 'sweating',
    'kambni': 'shivering',
    'behoshi': 'loss of consciousness',
    'dora': 'seizure',
    'joran da dard': 'joint pain',
    'kamar dard': 'back pain',
    'gala kharab': 'sore throat',
    'naak vagna': 'runny nose',
    'khoon': 'bleeding',
    'sar te chot': 'head injury',
    'haddi tut gayi': 'fracture',
    'gardan akri hoi': 'stiff neck',
    'tez bukhaar': 'high fever',
    'jorra sujna': 'joint swelling',
    'chehra latakna': 'facial drooping',
    'hath kamzor': 'arm weakness',
    'bolan vich mushkil': 'slurred speech',
    'khudkushi de khyal': 'suicidal thoughts',
    'peshab vich jalan': 'burning urination',
    'vaar vaar peshab': 'frequent urination',
    'lattan vich soojan': 'leg swelling',
    'lattan thandiyan': 'cold feet',
    'hath sunn': 'numbness in hands',
    'pair sunn': 'numbness in feet',
    'kaamp': 'tremor',
    'khoon wali khansi': 'coughing blood',
    'raat nu paseena': 'night sweats',
    'seene vich jalan': 'heartburn',
    'nigalan vich mushkil': 'difficulty swallowing',
    'malashe cho khoon': 'rectal bleeding',
    'tez pet dard': 'severe abdominal pain',
    'moonh vich chhale': 'mouth sores',
    'bhaunkan wali khansi': 'barking cough',
    'awaaz baith gayi': 'hoarse voice',
    'ghabrahat de daure': 'panic attacks',
    'kaamp lagna': 'trembling',
    'maaray khwab': 'nightmares',
    'khudkushi': 'suicidal thoughts',
    'khud kushi': 'suicidal thoughts',
}


#  Emergency First-Aid Instructions 

FIRST_AID_INSTRUCTIONS = {
    "chest pain": {
        "en": [
            "Loosen any tight clothing around the chest and neck.",
            "Keep the patient calm and seated — avoid physical exertion.",
            "Ensure fresh airflow; open a window or use a fan.",
            "If aspirin is available and patient is not allergic, give 325 mg to chew.",
            "Do NOT give food or water.",
            "Call emergency services (1122 / 115) immediately.",
        ],
        "ur": [
            "سینے اور گردن کے ارد گرد تنگ کپڑے ڈھیلے کریں۔",
            "مریض کو پرسکون رکھیں — جسمانی محنت سے بچائیں۔",
            "تازہ ہوا کا بندوبست کریں — کھڑکی کھولیں۔",
            "اگر اسپرین موجود ہو اور الرجی نہ ہو تو چبانے کو دیں۔",
            "کھانا یا پانی نہ دیں۔",
            "فوری طور پر ایمرجنسی (1122) کو کال کریں۔",
        ],
        "pa": [
            "سینے تے گلے دے آلے دوالے تنگ کپڑے ڈھلے کرو۔",
            "مریض نوں پرسکون رکھو — محنت توں بچاؤ۔",
            "تازی ہوا دا بندوبست کرو۔",
            "فوری طور تے 1122 نوں فون کرو۔",
        ],
    },
    "stroke": {
        "en": [
            "Act FAST: Face drooping? Arm weakness? Speech difficulty? Time to call!",
            "Keep the patient lying down with head slightly elevated.",
            "Do NOT give anything to eat or drink.",
            "Note the time symptoms started — critical for treatment.",
            "Call 1122 immediately.",
        ],
        "ur": [
            "FAST چیک کریں: چہرہ ٹیڑھا؟ ہاتھ کمزور؟ بولنے میں مشکل؟",
            "مریض کو لٹائیں، سر تھوڑا اونچا رکھیں۔",
            "کچھ کھلائیں یا پلائیں نہیں۔",
            "علامات شروع ہونے کا وقت نوٹ کریں۔",
            "فوری 1122 کال کریں۔",
        ],
        "pa": [
            "FAST: چہرہ ٹیڑھا، ہتھ کمزور، بولن وچ مشکل؟",
            "مریض نوں لٹاؤ، سر تھوڑا اچا رکھو۔",
            "کجھ نہ کھواؤ یا پیاؤ۔",
            "1122 نوں فوری کال کرو۔",
        ],
    },
    "seizure": {
        "en": [
            "Clear the area of hard/sharp objects around the patient.",
            "Gently turn the patient on their side to prevent choking.",
            "Do NOT put anything in the patient's mouth.",
            "Do NOT restrain or hold the patient down.",
            "Time the seizure — if it lasts more than 5 minutes, call 1122.",
            "Stay calm and stay with the patient.",
        ],
        "ur": [
            "مریض کے ارد گرد سخت چیزیں ہٹائیں۔",
            "مریض کو کروٹ پر لٹائیں تاکہ گلا نہ بھرے۔",
            "منہ میں کچھ نہ ڈالیں۔",
            "مریض کو روکنے کی کوشش نہ کریں۔",
            "دورے کا وقت دیکھیں — 5 منٹ سے زیادہ ہو تو 1122 کال کریں۔",
        ],
        "pa": [
            "مریض دے آلے دوالے چیزاں ہٹاؤ۔",
            "مریض نوں کروٹ تے لٹاؤ۔",
            "منہ وچ کجھ نہ پاؤ۔",
            "5 منٹ توں ودھ ہووے تے 1122 کال کرو۔",
        ],
    },
    "cannot breathe": {
        "en": [
            "Help the patient sit upright — do NOT lay them flat.",
            "Loosen tight clothing around chest and neck.",
            "If the patient uses an inhaler, help them use it immediately.",
            "Ensure fresh air — open windows/doors.",
            "Call 1122 immediately.",
        ],
        "ur": [
            "مریض کو سیدھا بیٹھائیں — لٹائیں نہیں۔",
            "سینے اور گردن کے کپڑے ڈھیلے کریں۔",
            "اگر انہیلر ہے تو فوراً استعمال کرائیں۔",
            "تازہ ہوا کا بندوبست کریں۔",
            "فوری 1122 کال کریں۔",
        ],
        "pa": [
            "مریض نوں سیدھا بٹھاؤ۔",
            "کپڑے ڈھلے کرو۔",
            "انہیلر ہووے تے ورتاؤ۔",
            "1122 نوں فوری کال کرو۔",
        ],
    },
    "loss of consciousness": {
        "en": [
            "Check if the patient is breathing.",
            "If not breathing, begin CPR immediately if trained.",
            "Place in recovery position (on side) if breathing.",
            "Do NOT give water or food.",
            "Call 1122 immediately.",
        ],
        "ur": [
            "چیک کریں کہ مریض سانس لے رہا ہے یا نہیں۔",
            "اگر سانس نہ ہو اور آپ جانتے ہوں تو CPR کریں۔",
            "سانس ہو تو کروٹ پر لٹائیں۔",
            "پانی یا کھانا نہ دیں۔",
            "فوری 1122 کال کریں۔",
        ],
        "pa": [
            "چیک کرو کہ سانس آ رہی اے۔",
            "سانس نہ ہووے تے CPR کرو۔",
            "کروٹ تے لٹاؤ۔",
            "1122 نوں فوری فون کرو۔",
        ],
    },
    "default": {
        "en": [
            "Keep the patient calm and in a comfortable position.",
            "Loosen any restrictive clothing.",
            "Monitor vital signs — breathing, pulse, consciousness.",
            "Do NOT give medication unless prescribed.",
            "Call 1122 or go to the nearest emergency room immediately.",
        ],
        "ur": [
            "مریض کو پرسکون رکھیں۔",
            "تنگ کپڑے ڈھیلے کریں۔",
            "سانس، نبض، ہوش چیک کرتے رہیں۔",
            "بغیر ڈاکٹر کے دوا نہ دیں۔",
            "فوری 1122 کال کریں یا قریبی ہسپتال جائیں۔",
        ],
        "pa": [
            "مریض نوں پرسکون رکھو۔",
            "تنگ کپڑے ڈھلے کرو۔",
            "سانس تے نبض چیک کردے رہو۔",
            "1122 نوں فوری کال کرو۔",
        ],
    },
}


#  Emergency Popup Messages 

EMERGENCY_POPUP_MESSAGES = {
    "en": "⚠️ Emergency Detected! Please visit the nearest hospital immediately. Call 1122 now.",
    "ur": "⚠️ ایمرجنسی! فوری قریبی ہسپتال جائیں۔ ابھی 1122 پر کال کریں۔",
    "pa": "⚠️ ایمرجنسی! فوری نزدیکی ہسپتال جاؤ۔ ہن 1122 تے کال کرو۔",
    "ar": "⚠️ طوارئ! توجه إلى أقرب مستشفى فوراً.",
}


#  Chat Response Templates 

RESPONSE_TEMPLATES = {
    "analysis_intro": {
        "en": "I've analysed your symptoms. Here's what I found:",
        "ur": "میں نے آپ کی علامات کا تجزیہ کیا۔ یہ نتائج ہیں:",
        "pa": "میں تہاڈی علامات دا تجزیہ کیتا۔ نتیجے ایہہ نے:",
    },
    "emergency_intro": {
        "en": "⚠️ Your symptoms indicate a possible emergency. Follow these first-aid steps immediately:",
        "ur": "⚠️ آپ کی علامات ایمرجنسی کی نشاندہی کرتی ہیں۔ فوری یہ اقدامات کریں:",
        "pa": "⚠️ تہاڈی علامات ایمرجنسی دسدیاں نے۔ فوری ایہہ قدم چکو:",
    },
    "severity_labels": {
        "mild": {"en": "Mild", "ur": "ہلکی", "pa": "ہلکی"},
        "moderate": {"en": "Moderate", "ur": "درمیانی", "pa": "درمیانی"},
        "emergency": {"en": "⚠️ Emergency", "ur": "⚠️ ایمرجنسی", "pa": "⚠️ ایمرجنسی"},
    },
    "see_specialist": {
        "en": "We recommend seeing a specialist. Review the recommended doctors below.",
        "ur": "ہم تجویز کرتے ہیں کہ آپ ماہر ڈاکٹر سے ملیں۔ نیچے تجویز کردہ ڈاکٹر دیکھیں۔",
        "pa": "اسیں صلاح دیندے آں کہ تسیں ماہر ڈاکٹر کول جاؤ۔",
    },
    "no_symptoms": {
        "en": "I couldn't detect specific symptoms. Please describe how you feel in more detail.",
        "ur": "میں علامات نہیں پہچان سکا۔ براہ کرم تفصیل سے بتائیں آپ کیسا محسوس کر رہے ہیں۔",
        "pa": "میں علامات نہیں پہچان سکیا۔ کرپا کرکے ہور تفصیل دیو۔",
    },
    "small_talk_greeting": {
        "en": "Hi! I'm the AlphaCare assistant. Tell me what symptoms you're experiencing - for example 'headache and fever' or 'chest pain' - and I'll help figure out what might be going on.",
        "ur": "السلام علیکم! میں AlphaCare اسسٹنٹ ہوں۔ اپنی علامات بتائیں، مثلاً 'سر درد اور بخار' یا 'سینے میں درد'، میں مدد کروں گا۔",
        "pa": "سلام! میں AlphaCare اسسٹنٹ آں۔ اپنیاں علامات دسو، جیویں 'سر درد تے بخار' یا 'چھاتی وچ درد'۔",
    },
    "small_talk_how_are_you": {
        "en": "I'm doing well, thanks for asking! I'm here to help with your health concerns - how are you feeling today?",
        "ur": "میں ٹھیک ہوں، شکریہ! میں آپ کی صحت میں مدد کے لیے حاضر ہوں - آپ کیسا محسوس کر رہے ہیں؟",
        "pa": "میں ٹھیک آں، شکریہ! میں تہاڈی صحت وچ مدد لئی حاضر آں - تسیں کیویں محسوس کر رہے او؟",
    },
    "small_talk_thanks": {
        "en": "You're welcome! Let me know if you'd like to describe any symptoms.",
        "ur": "خوش آمدید! اگر کوئی علامات بتانی ہوں تو ضرور بتائیں۔",
        "pa": "خوش آمدید! جے کوئی علامات دسنیاں نے تے ضرور دسو۔",
    },
    "small_talk_bye": {
        "en": "Take care! Come back anytime you're not feeling well.",
        "ur": "اپنا خیال رکھیں! جب بھی طبیعت خراب ہو، دوبارہ آ جائیں۔",
        "pa": "اپنا خیال رکھو! جدوں وی طبیعت خراب ہووے، واپس آ جاؤ۔",
    },
    "small_talk_capability": {
        "en": "I analyze the symptoms you describe and suggest possible conditions plus the right type of specialist to see. Try typing something like 'fever and sore throat'.",
        "ur": "میں آپ کی بتائی گئی علامات کا تجزیہ کر کے ممکنہ بیماری اور مناسب ڈاکٹر تجویز کرتا ہوں۔ مثلاً 'بخار اور گلے کی خرابی' لکھ کر دیکھیں۔",
        "pa": "میں تہاڈیاں دسیاں علامات دا تجزیہ کر کے بیماری تے صحیح ڈاکٹر دی صلاح دیندا آں۔ جیویں 'بخار تے گلا خراب' لکھ کے ویکھو۔",
    },
    "small_talk_ok": {
        "en": "Alright! Whenever you're ready, just describe how you're feeling and I'll take a look.",
        "ur": "ٹھیک ہے! جب بھی تیار ہوں، اپنی حالت بتائیں، میں دیکھتا ہوں۔",
        "pa": "ٹھیک اے! جدوں وی تیار ہوو، اپنی حالت دسو، میں ویکھدا آں۔",
    },
}


#  Language Detection 

def detect_language(text: str) -> str:
    """
    Returns: 'ur' (Urdu/Roman Urdu), 'pa' (Punjabi), 'ar' (Arabic), 'en' (English/default)
    """
    # Arabic/Urdu Unicode range check
    urdu_char_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF')
    gurmukhi_count = sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')

    if gurmukhi_count > 0:
        return 'pa'

    # If significant Arabic-script characters, determine Urdu vs Arabic
    if urdu_char_count >= 2:
        # Arabic-specific chars that don't appear in Urdu
        arabic_only = sum(1 for c in text if c in 'ةى')
        return 'ar' if arabic_only > urdu_char_count * 0.3 else 'ur'

    # Roman Urdu / Punjabi detection: check directly against the real
    # translation vocabulary first (any word we can translate is also a
    # signal for what language it's in) - this keeps detection in sync
    # automatically as the vocabulary grows, instead of a small hardcoded list.
    text_lower = text.lower()

    punjabi_vocab_hits = sum(
        1 for kw in PUNJABI_SYMPTOM_MAP
        if kw.isascii() and kw not in URDU_SYMPTOM_MAP and kw in text_lower
    )
    urdu_vocab_hits = sum(
        1 for kw in URDU_SYMPTOM_MAP
        if kw.isascii() and kw in text_lower
    )

    if punjabi_vocab_hits >= 1:
        return 'pa'
    if urdu_vocab_hits >= 1:
        return 'ur'

    # Fallback: short connector-word heuristic, for messages that use only
    # generic conversational words without a specific symptom term.
    urdu_roman_hits = sum(1 for kw in [
        'dard', 'bukhaar', 'bukhar', 'khansi', 'sar dard', 'chakkar',
        'ulti', 'matli', 'kamzori', 'thakawat', 'takleef', 'beemar',
        'mujhe', 'meri', 'mera', 'hai', 'hoon', 'nahi', 'kuch', 'bhi',
        'pet', 'seene', 'sine', 'gala', 'peeth', 'joron', 'bukhaar',
        'sans', 'naak', 'dil', 'khoon', 'paseena', 'kampna',
    ] if kw in text_lower)

    punjabi_roman_hits = sum(1 for kw in [
        'chaati', 'dil dard', 'sir dard', 'bukhaar hoya', 'koi takleef',
        'menu', 'tenu', 'karda', 'honda', 'ohna',
    ] if kw in text_lower)

    if urdu_roman_hits >= 2:
        return 'ur'
    if punjabi_roman_hits >= 2:
        return 'pa'

    return 'en'


def translate_to_english(text: str, lang: str) -> str:
    """
    Translate symptom text from detected language to English for ML model.
    """
    if lang == 'en':
        return text

    translation_map = {}
    if lang in ('ur',):
        translation_map = URDU_SYMPTOM_MAP
    elif lang == 'pa':
        translation_map = {**PUNJABI_SYMPTOM_MAP, **URDU_SYMPTOM_MAP}

    result = text
    # Sort by length descending to replace longer phrases first
    for src, tgt in sorted(translation_map.items(), key=lambda x: -len(x[0])):
        result = result.replace(src, tgt)

    return result


def get_first_aid(symptoms_text: str, lang: str) -> list:
    """
    Return first-aid instructions in detected language based on symptoms.
    """
    text_lower = symptoms_text.lower()

    # Map symptom to first-aid category
    if any(k in text_lower for k in ['chest pain', 'seene mein dard', 'sine mein dard', 'سینے میں درد', 'سینے کا درد', 'heart attack']):
        key = "chest pain"
    elif any(k in text_lower for k in ['stroke', 'face drooping', 'arm weakness', 'slurred speech', 'منہ ٹیڑھا']):
        key = "stroke"
    elif any(k in text_lower for k in ['seizure', 'convulsion', 'دورہ']):
        key = "seizure"
    elif any(k in text_lower for k in ['cannot breathe', "can't breathe", 'severe breathing', 'سانس نہیں']):
        key = "cannot breathe"
    elif any(k in text_lower for k in ['unconscious', 'loss of consciousness', 'بے ہوشی', 'behoshi']):
        key = "loss of consciousness"
    else:
        key = "default"

    instructions = FIRST_AID_INSTRUCTIONS.get(key, FIRST_AID_INSTRUCTIONS["default"])
    return instructions.get(lang, instructions["en"])


def get_emergency_popup_message(lang: str) -> str:
    return EMERGENCY_POPUP_MESSAGES.get(lang, EMERGENCY_POPUP_MESSAGES["en"])


def get_template(key: str, lang: str, sub_key: str = None) -> str:
    """Get a response template in the detected language."""
    template = RESPONSE_TEMPLATES.get(key, {})
    if sub_key:
        template = template.get(sub_key, {})
    return template.get(lang, template.get("en", ""))




# """
# Multilingual NLP support for AlphaCare.
# Detects language from user input and provides:
#   - Symptom keyword translation to English for model
#   - First-aid instructions in detected language
#   - Emergency messages in detected language
# """

# #  Language Detection Patterns 

# # Urdu common words / characters
# URDU_PATTERNS = [
#     'درد', 'بخار', 'کھانسی', 'سردرد', 'سانس', 'چکر', 'الٹی', 'متلی',
#     'دل', 'سینہ', 'پیٹ', 'کمزوری', 'تھکاوٹ', 'بیمار', 'مرض', 'علامت',
#     'ہاں', 'نہیں', 'ہے', 'میں', 'مجھے', 'مریض', 'دوائی', 'ڈاکٹر',
#     'ہسپتال', 'کیا', 'ہوں', 'بہت', 'ٹھیک', 'تکلیف',
# ]

# # Punjabi common words
# PUNJABI_PATTERNS = [
#     'ਦਰਦ', 'ਬੁਖਾਰ', 'ਖੰਘ', 'ਸਿਰਦਰਦ', 'ਚੱਕਰ', 'ਉਲਟੀ',
#     'ਦਿਲ', 'ਛਾਤੀ', 'ਕਮਜ਼ੋਰੀ', 'ਥਕਾਵਟ', 'ਬਿਮਾਰ',
#     # Punjabi in Roman/Shahmukhi
#     'dard', 'bukhaar', 'khansi', 'sir dard', 'chakkar', 'ulti',
#     'thand', 'garmi', 'beemar', 'takleef', 'dil', 'chaati',
# ]

# # Arabic patterns (bonus)
# ARABIC_PATTERNS = [
#     'ألم', 'حمى', 'سعال', 'صداع', 'دوار', 'غثيان', 'قلب',
# ]


# #  Urdu Symptom Translations → English 

# URDU_SYMPTOM_MAP = {
#     # Pain
#     'درد': 'pain',
#     'سردرد': 'headache',
#     'سر درد': 'headache',
#     'سینے میں درد': 'chest pain',
#     'سینے کا درد': 'chest pain',
#     'پیٹ درد': 'abdominal pain',
#     'پیٹ میں درد': 'abdominal pain',
#     'کمر درد': 'back pain',
#     'جوڑوں کا درد': 'joint pain',
#     'گلے میں درد': 'sore throat',
#     'گلا خراب': 'sore throat',

#     # Fever/Temperature
#     'بخار': 'fever',
#     'تیز بخار': 'high fever',
#     'ہلکا بخار': 'mild fever',

#     # Respiratory
#     'کھانسی': 'cough',
#     'سانس لینے میں تکلیف': 'shortness of breath',
#     'سانس پھولنا': 'shortness of breath',
#     'ناک بہنا': 'runny nose',
#     'چھینکیں': 'sneezing',
#     'سانس نہیں آ رہی': 'cannot breathe',
#     'سانس نہیں آتی': 'cannot breathe',

#     # Digestive
#     'الٹی': 'vomiting',
#     'متلی': 'nausea',
#     'دست': 'diarrhea',
#     'قبض': 'constipation',
#     'بدہضمی': 'indigestion',

#     # General
#     'چکر': 'dizziness',
#     'چکر آنا': 'dizziness',
#     'کمزوری': 'weakness',
#     'تھکاوٹ': 'fatigue',
#     'تھکان': 'fatigue',
#     'نیند نہ آنا': 'insomnia',
#     'بھوک نہ لگنا': 'loss of appetite',
#     'وزن کم ہونا': 'weight loss',
#     'جلد پر دانے': 'skin rash',
#     'خارش': 'itching',
#     'دھڑکن تیز': 'rapid heartbeat',
#     'منہ خشک': 'dry mouth',
#     'آنکھیں پیلی': 'yellow eyes',
#     'یرقان': 'jaundice',
#     'پسینہ': 'sweating',
#     'کانپنا': 'shivering',
#     'دورہ': 'seizure',
#     'بے ہوشی': 'loss of consciousness',
#     'منہ ٹیڑھا': 'face drooping',
#     'ہاتھ کمزور': 'arm weakness',
#     'بولنے میں تکلیف': 'slurred speech',
#     'خودکشی': 'suicidal thoughts',
#     'خودکشی کے خیالات': 'suicidal thoughts',
#     'bijli ka jhatka': 'electric shock',

        
    
#     # Roman Urdu keywords
#     'dard': 'pain',
#     'bukhaar': 'fever',
#     'bukhar': 'fever',
#     'khansi': 'cough',
#     'sar dard': 'headache',
#     'seene mein dard': 'chest pain',
#     'sine mein dard': 'chest pain',
#     'ulti': 'vomiting',
#     'accident' :'emergency',
#     'matli': 'nausea',
#     'chakkar': 'dizziness',
#     'kamzori': 'weakness',
#     'thakawat': 'fatigue',
#     'sans lene mein takleef': 'shortness of breath',
#     'sans nahi aa rahi': 'cannot breathe',
#     'dast': 'diarrhea',
#     'khujli': 'itching',
#     'paseena': 'sweating',
#     'kampna': 'shivering',
#     'behoshi': 'loss of consciousness',
#     'dora': 'seizure',
#     'beemar': 'sick',
#     'takleef': 'discomfort',
#     'garmi': 'heat',
#     'thand': 'cold',
#     'joron ka dard': 'joint pain',
#     'peeth dard': 'back pain',
#     'pet dard': 'abdominal pain',
#     'gala kharab': 'sore throat',
#     'naak beh rahi': 'runny nose',
#     'khoon': 'bleeding',
#     'zeher': 'poison',
#     'jal gaya': 'burn',
#     'chot': 'injury',
#     'haddi toot': 'fracture',
# }

# # Punjabi Roman → English
# PUNJABI_SYMPTOM_MAP = {
#     'dard': 'pain',
#     'bukhaar': 'fever',
#     'khansi': 'cough',
#     'sir dard': 'headache',
#     'chakkar': 'dizziness',
#     'ulti': 'vomiting',
#     'kamzori': 'weakness',
#     'thakawat': 'fatigue',
#     'dil dard': 'chest pain',
#     'pet dard': 'abdominal pain',
#     'beemar': 'sick',
#     'takleef': 'discomfort',
# }


# #  Emergency First-Aid Instructions 

# FIRST_AID_INSTRUCTIONS = {
#     "chest pain": {
#         "en": [
#             "Loosen any tight clothing around the chest and neck.",
#             "Keep the patient calm and seated — avoid physical exertion.",
#             "Ensure fresh airflow; open a window or use a fan.",
#             "If aspirin is available and patient is not allergic, give 325 mg to chew.",
#             "Do NOT give food or water.",
#             "Call emergency services (1122 / 115) immediately.",
#         ],
#         "ur": [
#             "سینے اور گردن کے ارد گرد تنگ کپڑے ڈھیلے کریں۔",
#             "مریض کو پرسکون رکھیں — جسمانی محنت سے بچائیں۔",
#             "تازہ ہوا کا بندوبست کریں — کھڑکی کھولیں۔",
#             "اگر اسپرین موجود ہو اور الرجی نہ ہو تو چبانے کو دیں۔",
#             "کھانا یا پانی نہ دیں۔",
#             "فوری طور پر ایمرجنسی (1122) کو کال کریں۔",
#         ],
#         "pa": [
#             "سینے تے گلے دے آلے دوالے تنگ کپڑے ڈھلے کرو۔",
#             "مریض نوں پرسکون رکھو — محنت توں بچاؤ۔",
#             "تازی ہوا دا بندوبست کرو۔",
#             "فوری طور تے 1122 نوں فون کرو۔",
#         ],
#     },
#     "stroke": {
#         "en": [
#             "Act FAST: Face drooping? Arm weakness? Speech difficulty? Time to call!",
#             "Keep the patient lying down with head slightly elevated.",
#             "Do NOT give anything to eat or drink.",
#             "Note the time symptoms started — critical for treatment.",
#             "Call 1122 immediately.",
#         ],
#         "ur": [
#             "FAST چیک کریں: چہرہ ٹیڑھا؟ ہاتھ کمزور؟ بولنے میں مشکل؟",
#             "مریض کو لٹائیں، سر تھوڑا اونچا رکھیں۔",
#             "کچھ کھلائیں یا پلائیں نہیں۔",
#             "علامات شروع ہونے کا وقت نوٹ کریں۔",
#             "فوری 1122 کال کریں۔",
#         ],
#         "pa": [
#             "FAST: چہرہ ٹیڑھا، ہتھ کمزور، بولن وچ مشکل؟",
#             "مریض نوں لٹاؤ، سر تھوڑا اچا رکھو۔",
#             "کجھ نہ کھواؤ یا پیاؤ۔",
#             "1122 نوں فوری کال کرو۔",
#         ],
#     },
#     "seizure": {
#         "en": [
#             "Clear the area of hard/sharp objects around the patient.",
#             "Gently turn the patient on their side to prevent choking.",
#             "Do NOT put anything in the patient's mouth.",
#             "Do NOT restrain or hold the patient down.",
#             "Time the seizure — if it lasts more than 5 minutes, call 1122.",
#             "Stay calm and stay with the patient.",
#         ],
#         "ur": [
#             "مریض کے ارد گرد سخت چیزیں ہٹائیں۔",
#             "مریض کو کروٹ پر لٹائیں تاکہ گلا نہ بھرے۔",
#             "منہ میں کچھ نہ ڈالیں۔",
#             "مریض کو روکنے کی کوشش نہ کریں۔",
#             "دورے کا وقت دیکھیں — 5 منٹ سے زیادہ ہو تو 1122 کال کریں۔",
#         ],
#         "pa": [
#             "مریض دے آلے دوالے چیزاں ہٹاؤ۔",
#             "مریض نوں کروٹ تے لٹاؤ۔",
#             "منہ وچ کجھ نہ پاؤ۔",
#             "5 منٹ توں ودھ ہووے تے 1122 کال کرو۔",
#         ],
#     },
#     "cannot breathe": {
#         "en": [
#             "Help the patient sit upright — do NOT lay them flat.",
#             "Loosen tight clothing around chest and neck.",
#             "If the patient uses an inhaler, help them use it immediately.",
#             "Ensure fresh air — open windows/doors.",
#             "Call 1122 immediately.",
#         ],
#         "ur": [
#             "مریض کو سیدھا بیٹھائیں — لٹائیں نہیں۔",
#             "سینے اور گردن کے کپڑے ڈھیلے کریں۔",
#             "اگر انہیلر ہے تو فوراً استعمال کرائیں۔",
#             "تازہ ہوا کا بندوبست کریں۔",
#             "فوری 1122 کال کریں۔",
#         ],
#         "pa": [
#             "مریض نوں سیدھا بٹھاؤ۔",
#             "کپڑے ڈھلے کرو۔",
#             "انہیلر ہووے تے ورتاؤ۔",
#             "1122 نوں فوری کال کرو۔",
#         ],
#     },
#     "loss of consciousness": {
#         "en": [
#             "Check if the patient is breathing.",
#             "If not breathing, begin CPR immediately if trained.",
#             "Place in recovery position (on side) if breathing.",
#             "Do NOT give water or food.",
#             "Call 1122 immediately.",
#         ],
#         "ur": [
#             "چیک کریں کہ مریض سانس لے رہا ہے یا نہیں۔",
#             "اگر سانس نہ ہو اور آپ جانتے ہوں تو CPR کریں۔",
#             "سانس ہو تو کروٹ پر لٹائیں۔",
#             "پانی یا کھانا نہ دیں۔",
#             "فوری 1122 کال کریں۔",
#         ],
#         "pa": [
#             "چیک کرو کہ سانس آ رہی اے۔",
#             "سانس نہ ہووے تے CPR کرو۔",
#             "کروٹ تے لٹاؤ۔",
#             "1122 نوں فوری فون کرو۔",
#         ],
#     },
#     "default": {
#         "en": [
#             "Keep the patient calm and in a comfortable position.",
#             "Loosen any restrictive clothing.",
#             "Monitor vital signs — breathing, pulse, consciousness.",
#             "Do NOT give medication unless prescribed.",
#             "Call 1122 or go to the nearest emergency room immediately.",
#         ],
#         "ur": [
#             "مریض کو پرسکون رکھیں۔",
#             "تنگ کپڑے ڈھیلے کریں۔",
#             "سانس، نبض، ہوش چیک کرتے رہیں۔",
#             "بغیر ڈاکٹر کے دوا نہ دیں۔",
#             "فوری 1122 کال کریں یا قریبی ہسپتال جائیں۔",
#         ],
#         "pa": [
#             "مریض نوں پرسکون رکھو۔",
#             "تنگ کپڑے ڈھلے کرو۔",
#             "سانس تے نبض چیک کردے رہو۔",
#             "1122 نوں فوری کال کرو۔",
#         ],
#     },
# }


# #  Emergency Popup Messages 

# EMERGENCY_POPUP_MESSAGES = {
#     "en": "⚠️ Emergency Detected! Please visit the nearest hospital immediately. Call 1122 now.",
#     "ur": "⚠️ ایمرجنسی! فوری قریبی ہسپتال جائیں۔ ابھی 1122 پر کال کریں۔",
#     "pa": "⚠️ ایمرجنسی! فوری نزدیکی ہسپتال جاؤ۔ ہن 1122 تے کال کرو۔",
#     "ar": "⚠️ طوارئ! توجه إلى أقرب مستشفى فوراً.",
# }


# #  Chat Response Templates 

# RESPONSE_TEMPLATES = {
#     "analysis_intro": {
#         "en": "I've analysed your symptoms. Here's what I found:",
#         "ur": "میں نے آپ کی علامات کا تجزیہ کیا۔ یہ نتائج ہیں:",
#         "pa": "میں تہاڈی علامات دا تجزیہ کیتا۔ نتیجے ایہہ نے:",
#     },
#     "emergency_intro": {
#         "en": "⚠️ Your symptoms indicate a possible emergency. Follow these first-aid steps immediately:",
#         "ur": "⚠️ آپ کی علامات ایمرجنسی کی نشاندہی کرتی ہیں۔ فوری یہ اقدامات کریں:",
#         "pa": "⚠️ تہاڈی علامات ایمرجنسی دسدیاں نے۔ فوری ایہہ قدم چکو:",
#     },
#     "severity_labels": {
#         "mild": {"en": "Mild", "ur": "ہلکی", "pa": "ہلکی"},
#         "moderate": {"en": "Moderate", "ur": "درمیانی", "pa": "درمیانی"},
#         "emergency": {"en": "⚠️ Emergency", "ur": "⚠️ ایمرجنسی", "pa": "⚠️ ایمرجنسی"},
#     },
#     "see_specialist": {
#         "en": "We recommend seeing a specialist. Review the recommended doctors below.",
#         "ur": "ہم تجویز کرتے ہیں کہ آپ ماہر ڈاکٹر سے ملیں۔ نیچے تجویز کردہ ڈاکٹر دیکھیں۔",
#         "pa": "اسیں صلاح دیندے آں کہ تسیں ماہر ڈاکٹر کول جاؤ۔",
#     },
#     "no_symptoms": {
#         "en": "I couldn't detect specific symptoms. Please describe how you feel in more detail.",
#         "ur": "میں علامات نہیں پہچان سکا۔ براہ کرم تفصیل سے بتائیں آپ کیسا محسوس کر رہے ہیں۔",
#         "pa": "میں علامات نہیں پہچان سکیا۔ کرپا کرکے ہور تفصیل دیو۔",
#     },
# }


# #  Language Detection 

# def detect_language(text: str) -> str:
#     """
#     Returns: 'ur' (Urdu/Roman Urdu), 'pa' (Punjabi), 'ar' (Arabic), 'en' (English/default)
#     """
#     # Arabic/Urdu Unicode range check
#     urdu_char_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF')
#     gurmukhi_count = sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')

#     if gurmukhi_count > 0:
#         return 'pa'

#     # If significant Arabic-script characters, determine Urdu vs Arabic
#     if urdu_char_count >= 2:
#         # Arabic-specific chars that don't appear in Urdu
#         arabic_only = sum(1 for c in text if c in 'ةى')
#         return 'ar' if arabic_only > urdu_char_count * 0.3 else 'ur'

#     # Roman Urdu / Punjabi detection via keyword matching
#     text_lower = text.lower()
#     urdu_roman_hits = sum(1 for kw in [
#         'dard', 'bukhaar', 'bukhar', 'khansi', 'sar dard', 'chakkar',
#         'ulti', 'matli', 'kamzori', 'thakawat', 'takleef', 'beemar',
#         'mujhe', 'meri', 'mera', 'hai', 'hoon', 'nahi', 'kuch', 'bhi',
#         'pet', 'seene', 'sine', 'gala', 'peeth', 'joron', 'bukhaar',
#         'sans', 'naak', 'dil', 'khoon', 'paseena', 'kampna',
#     ] if kw in text_lower)

#     punjabi_roman_hits = sum(1 for kw in [
#         'chaati', 'dil dard', 'sir dard', 'bukhaar hoya', 'koi takleef',
#         'menu', 'tenu', 'karda', 'honda', 'ohna',
#     ] if kw in text_lower)

#     if urdu_roman_hits >= 2:
#         return 'ur'
#     if punjabi_roman_hits >= 2:
#         return 'pa'

#     return 'en'


# def translate_to_english(text: str, lang: str) -> str:
#     """
#     Translate symptom text from detected language to English for ML model.
#     """
#     if lang == 'en':
#         return text

#     translation_map = {}
#     if lang in ('ur',):
#         translation_map = URDU_SYMPTOM_MAP
#     elif lang == 'pa':
#         translation_map = {**PUNJABI_SYMPTOM_MAP, **URDU_SYMPTOM_MAP}

#     result = text
#     # Sort by length descending to replace longer phrases first
#     for src, tgt in sorted(translation_map.items(), key=lambda x: -len(x[0])):
#         result = result.replace(src, tgt)

#     return result


# def get_first_aid(symptoms_text: str, lang: str) -> list:
#     """
#     Return first-aid instructions in detected language based on symptoms.
#     """
#     text_lower = symptoms_text.lower()

#     # Map symptom to first-aid category
#     if any(k in text_lower for k in ['chest pain', 'seene mein dard', 'sine mein dard', 'سینے میں درد', 'سینے کا درد', 'heart attack']):
#         key = "chest pain"
#     elif any(k in text_lower for k in ['stroke', 'face drooping', 'arm weakness', 'slurred speech', 'منہ ٹیڑھا']):
#         key = "stroke"
#     elif any(k in text_lower for k in ['seizure', 'convulsion', 'دورہ']):
#         key = "seizure"
#     elif any(k in text_lower for k in ['cannot breathe', "can't breathe", 'severe breathing', 'سانس نہیں']):
#         key = "cannot breathe"
#     elif any(k in text_lower for k in ['unconscious', 'loss of consciousness', 'بے ہوشی', 'behoshi']):
#         key = "loss of consciousness"
#     else:
#         key = "default"

#     instructions = FIRST_AID_INSTRUCTIONS.get(key, FIRST_AID_INSTRUCTIONS["default"])
#     return instructions.get(lang, instructions["en"])


# def get_emergency_popup_message(lang: str) -> str:
#     return EMERGENCY_POPUP_MESSAGES.get(lang, EMERGENCY_POPUP_MESSAGES["en"])


# def get_template(key: str, lang: str, sub_key: str = None) -> str:
#     """Get a response template in the detected language."""
#     template = RESPONSE_TEMPLATES.get(key, {})
#     if sub_key:
#         template = template.get(sub_key, {})
#     return template.get(lang, template.get("en", ""))

