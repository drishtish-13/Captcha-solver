import os
import random
import string
import pyttsx3

output_folder = "data/audio"
os.makedirs(output_folder, exist_ok=True)

engine = pyttsx3.init()

engine.setProperty('rate', 150)  
engine.setProperty('volume', 1.0)  # Volume 0-1

def generate_random_captcha(length=6):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def save_captcha_audio(text, filename):
    engine.save_to_file(text, filename)
    engine.runAndWait()

def main(num_samples=50):
    labels_path = os.path.join(output_folder, "labels.csv")
    with open(labels_path, "w") as f:
        for i in range(1, num_samples + 1):
            captcha_text = generate_random_captcha()
            filename = f"captcha_{i}.wav"
            filepath = os.path.join(output_folder, filename)
            print(f"Generating audio for captcha: {captcha_text} -> {filepath}")
            save_captcha_audio(captcha_text, filepath)
            f.write(f"{filename},{captcha_text}\n")

if __name__ == "__main__":
    main()
