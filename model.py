import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = None
wav2vec = None
classifier = None


def load_models():
    global processor, wav2vec, classifier

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base"
    )

    wav2vec = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base"
    )
    wav2vec.eval()

    classifier = torch.load(
        "classifier.pt",
        map_location="cpu"
    )
    classifier.eval()


@torch.no_grad()
def predict_ai(audio):
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    outputs = wav2vec(**inputs)
    features = outputs.last_hidden_state.mean(dim=1)

    prob = torch.sigmoid(classifier(features)).item()
    return prob
