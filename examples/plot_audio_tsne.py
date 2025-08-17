import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import WhisperModel, WhisperFeatureExtractor, Wav2Vec2BertModel, AutoFeatureExtractor
import librosa
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

AUDIO_FOLDER = "audio"
OUTPUT_FOLDER = "plots"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3").to(device)
whisper_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

w2v_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to(device)
w2v_processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

Path(OUTPUT_FOLDER).mkdir(exist_ok=True)


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    return audio


def extract_embeddings(model, processor, audio_files, model_name):
    all_embeddings_last = []
    all_embeddings_third = []
    labels = []

    for file_path, language in audio_files:
        audio = load_audio(file_path)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)

        with torch.no_grad():
            if model_name == "whisper":
                outputs = model.encoder(**inputs, output_hidden_states=True)
            else:
                outputs = model(**inputs, output_hidden_states=True)

            last_hidden = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
            third_last_hidden = outputs.hidden_states[-3].mean(dim=1).cpu().numpy()

        all_embeddings_last.append(last_hidden[0])
        all_embeddings_third.append(third_last_hidden[0])
        labels.append(language)

    return np.array(all_embeddings_last), np.array(all_embeddings_third), labels


def plot_tsne(embeddings, labels, title, filename):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(14, 10))
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[colors[i]], label=label, s=100, alpha=0.7)

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()


audio_files = []
for lang_folder in Path(AUDIO_FOLDER).iterdir():
    if lang_folder.is_dir():
        language = lang_folder.name
        for audio_file in lang_folder.glob("*.wav"):
            audio_files.append((str(audio_file), language))

whisper_last, whisper_third, labels = extract_embeddings(whisper_model, whisper_processor, audio_files, "whisper")
w2v_last, w2v_third, _ = extract_embeddings(w2v_model, w2v_processor, audio_files, "w2v")

plot_tsne(whisper_last, labels, "Whisper Large v3 - Last Hidden State", "whisper_last_hidden.png")
plot_tsne(whisper_third, labels, "Whisper Large v3 - Third-to-Last Hidden State", "whisper_third_last.png")
plot_tsne(w2v_last, labels, "W2V-BERT 2.0 - Last Hidden State", "w2v_bert_last_hidden.png")
plot_tsne(w2v_third, labels, "W2V-BERT 2.0 - Third-to-Last Hidden State", "w2v_bert_third_last.png")

print(f"Plots saved to {OUTPUT_FOLDER}/")