import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Wav2Vec2 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model_asr = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)

def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    audio = librosa.util.normalize(audio)  # Normalize
    return audio

# Convert audio to text using Wav2Vec2
def transcribe_audio(file_path):
    audio = load_audio(file_path)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        logits = model_asr(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.strip()

vocab = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")}
vocab["<unk>"] = len(vocab)

def pad_sequence(sequence, max_length):
    return F.pad(sequence, (0, max_length - len(sequence)), value=0)

# Dataset class (Audio only)
class CaptchaAudioDataset(Dataset):
    def __init__(self, audio_dir, vocab, max_audio_len=100):
        self.audio_files = sorted(os.listdir(audio_dir))
        self.audio_dir = audio_dir
        self.vocab = vocab
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.audio_files)

    def text_to_sequence(self, text):
        return [self.vocab.get(char, self.vocab["<unk>"]) for char in text]

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        audio_text = transcribe_audio(audio_path)
        audio_sequence = self.text_to_sequence(audio_text)
        audio_sequence = pad_sequence(torch.tensor(audio_sequence, dtype=torch.float32), self.max_audio_len)
        audio_sequence = audio_sequence.unsqueeze(0)
        return audio_sequence, audio_sequence  # Using input as target too

# CRNN model
class CRNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_classes=len(vocab)):
        super(CRNN, self).__init__()
        self.cnn = nn.Conv1d(1, 100, kernel_size=3, padding=1)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def main():
    model = CRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = CaptchaAudioDataset("data/audio", vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        start_time = time.time()

        for audio_text, target_text in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            optimizer.zero_grad()
            audio_text = audio_text.to(device)
            target_text = target_text.to(device).long()

            outputs = model(audio_text)

            batch_size = outputs.shape[0]
            seq_length = outputs.shape[1]
            num_classes = outputs.shape[2]

            target_text = pad_sequence(target_text.view(-1), seq_length * batch_size).view(batch_size, seq_length)

            outputs = outputs.view(-1, num_classes)
            target_text = target_text.view(-1)

            loss = criterion(outputs, target_text)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == target_text).sum().item()
            total += target_text.size(0)

        train_accuracy = (correct / total) * 100
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Time: {epoch_time:.2f}s")
        torch.save(model.state_dict(), f'model_checkpoint_epoch_{epoch+1}.pth')

    # Final evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for audio_text, target_text in dataloader:
            audio_text = audio_text.to(device)
            target_text = target_text.to(device).squeeze(1).long()
            outputs = model(audio_text)
            predicted = torch.argmax(outputs, dim=2)
            correct += (predicted == target_text).sum().item()
            total += target_text.size(0) * target_text.size(1)

    accuracy = (correct / total) * 100
    print(f"Final Accuracy: {accuracy:.2f}%")
    torch.save(model.state_dict(), 'final_model.pth')

if __name__ == "__main__":
    main()
