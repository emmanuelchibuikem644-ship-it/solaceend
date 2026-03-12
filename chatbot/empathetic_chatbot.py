from inference.emotion_predictor import predict_emotion
from inference.response_generator import generate_response
from safety.safety_filter import is_safe


def chatbot_response(user_message):

    emotion = predict_emotion(user_message)

    response = generate_response(user_message)

    if not is_safe(response):

        response = "I'm here to support you. Tell me more about how you're feeling."

    return {
        "emotion": emotion,
        "response": response
    }