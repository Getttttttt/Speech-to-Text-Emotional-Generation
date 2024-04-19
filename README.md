# Speech-to-Text-Emotional-Generation
This Python-based project leverages advanced machine learning techniques to generate speech underlying emotions, enhancing user experience in applications ranging from virtual assistants to interactive gaming. Dive into our code, experiment with different emotional tones, and contribute to making digital communication more human-like!

## 实现思路

通过 识别-生成 两步完成

### Step 1: 文本情感识别（已调试通过）

这一步骤的目标是从给定的文本中自动识别出其中的情感类型以及情感的强度。利用预训练的情感分析模型实现，模型会接收输入的文本，并返回一个包含情感类别（如joy）和一个代表情感强度的数值（介于0到1之间的概率值，值越大表示情感表现越明显）。

定义了函数`classify_text_emotion`，接受一段文本作为输入，调用模型，然后返回情感类别和强度。这为下一步的语音生成准备了情感信息。

### Step 2: 文本到语音的情感生成（已完成但未调试）

在成功提取出文本的情感类型和强度后，下一步是将这些情感信息转化为语音，该工具支持在不同说话者之间转移情感，以生成富有表现力的语音。

具体实现过程中，根据从第一步得到的情感类型和强度，选择适当的说话者ID和情感ID。然后，运行命令行脚本`synthesize.py`，该脚本根据提供的文本、说话者ID、情感ID等参数，合成并输出带有特定情感的语音文件。

## Step1 文本情感识别

Ref: https://huggingface.co/michellejieli/emotion_text_classifier

```Python
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
```

提取出Text的Emotion Type和Intensity

## Step2 Speech-to-Text

Ref: https://github.com/keonlee9420/Cross-Speaker-Emotion-Transfer

将Step1中提取的结果作为Step2的input项输入，使用语音合成生成带有情感的语音。(环境配置参考Ref)

GitHub中的运行示例为：

```Bash
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --speaker_id SPEAKER_ID --emotion_id EMOTION_ID --restore_step RESTORE_STEP --mode single --dataset DATASET
```

输出音频文件

基于此给出的代码为：

```Python
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
```
