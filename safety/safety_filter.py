from safety.crisis_detector import detect_crisis


UNSAFE_WORDS = [
    "hate",
    "stupid",
    "idiot"
    
]


def is_safe(text):

    text = text.lower()

    for word in UNSAFE_WORDS:
        if word in text:
            return False

    return True


def safe_response(user_input, bot_response):

    # Crisis detection
    if detect_crisis(user_input):

        return (
            "I'm really sorry you're feeling this way. "
            "You don't have to go through it alone. "
            "Please consider reaching out to a trusted person or a professional for support."
        )

    # Unsafe chatbot output
    if not is_safe(bot_response):

        return (
            "I'm here to support you. "
            "Could you tell me more about how you're feeling?"
        )

    return bot_response