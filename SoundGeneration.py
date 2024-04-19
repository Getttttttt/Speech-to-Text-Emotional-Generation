import subprocess
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

def classify_text_emotion(text):
    results = classifier(text)
    emotion_type = results[0]['label']
    emotion_intensity = results[0]['score']
    return emotion_type, emotion_intensity

def synthesize_emotional_speech(text, speaker_id, emotion_id, restore_step, dataset):
    command = [
        'python3', './Cross-Speaker-Emotion-Transfer/synthesize.py',
        '--text', text,
        '--speaker_id', str(speaker_id),
        '--emotion_id', str(emotion_id),
        '--restore_step', str(restore_step),
        '--mode', 'single',
        '--dataset', dataset
    ]
    subprocess.run(command)

if __name__ == '__main__':
    text = 'I love life!'
    emotion_type, emotion_intensity = classify_text_emotion(text)
    '''
    由于时间限制，这一部分两边的output和input还没有对齐，但是大致的逻辑是这样的
    '''
    speaker_id = 1
    emotion_id = 3  
    restore_step = 100000
    dataset = 'your_dataset_name'
    synthesize_emotional_speech(text, speaker_id, emotion_id, restore_step, dataset)
