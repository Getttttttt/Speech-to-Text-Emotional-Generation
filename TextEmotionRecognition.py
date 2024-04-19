from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

def classify_text_emotion(text):
    results = classifier(text)
    emotion_type = results[0]['label']
    emotion_intensity = results[0]['score']
    return emotion_type, emotion_intensity

if __name__ == '__main__':
    text = 'I love life!'
    emotion_type, emotion_intensity = classify_text_emotion(text)
    print(f"Emotion Type: {emotion_type}, Intensity: {emotion_intensity}")
