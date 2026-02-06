import os
import librosa
import numpy as np
import torch
from torch import nn

DATA_DIR = "dataset"
SR = 16000


def extract_features(path):
    audio, _ = librosa.load(path, sr=SR)
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.var(axis=1)])
    return feat


X, y = [], []

for label, cls in enumerate(["human", "ai"]):
    folder = os.path.join(DATA_DIR, cls)
    for file in os.listdir(folder):
        if not file.lower().endswith(".wav"):
            continue  # skip non-audio files

        feat = extract_features(os.path.join(folder, file))
        X.append(feat)
        y.append(label)


X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

model = nn.Sequential(
    nn.Linear(40, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

torch.save(model, "classifier.pt")
print("Saved classifier.pt")

