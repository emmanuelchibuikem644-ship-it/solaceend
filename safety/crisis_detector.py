CRISIS_KEYWORDS = [
    "suicide",
    "kill myself",
    "end my life",
    "i want to die",
    "self harm",
    "hurt myself"
]

def detect_crisis(text):

    text = text.lower()

    for word in CRISIS_KEYWORDS:
        if word in text:
            return True

    return False