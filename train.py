import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

X, y = [], []

def load_folder(folder, label):
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        audio, sr = librosa.load(path, sr=16000)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze()

        X.append(emb)
        y.append(label)

load_folder("dataset/human", 0)
load_folder("dataset/ai", 1)

X = torch.stack(X)
y = torch.tensor(y).float().unsqueeze(1)

clf = torch.nn.Linear(768, 1)
optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(30):
    optimizer.zero_grad()
    logits = clf(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(clf.state_dict(), "classifier.pt")
print("Saved classifier.pt")
