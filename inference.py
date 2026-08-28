import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models

# ==============================
# PATHS
# ==============================
save_dir = "./saved_models"

mid_weights_path = f"{save_dir}/mid_fusion_best_weights.pth"
late_weights_path = f"{save_dir}/late_fusion_best_weights.pth"
tokenizer_path = f"{save_dir}/tokenizer"

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD TOKENIZER
# ==============================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# ==============================
# IMAGE TRANSFORM (MATCH PREPROCESSING)
# ==============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()  # already 0–1 since images were saved as uint8
])

# ==============================
# MODEL DEFINITIONS (EXACT MATCH)
# ==============================

class MidFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # MRI branch
        self.cnn = models.resnet18(weights=None)  # IMPORTANT: no pretrained at inference
        self.cnn.fc = nn.Linear(512, 256)

        # Text branch
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_fc = nn.Linear(768, 256)

        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, img, input_ids, attention_mask):
        img_feat = self.cnn(img)

        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        text_feat = self.text_fc(text_feat)

        fused = torch.cat((img_feat, text_feat), dim=1)
        return self.classifier(fused)


class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # MRI branch
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Linear(512, 1)

        # Text branch
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_fc = nn.Linear(768, 1)

        # Learnable weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, img, input_ids, attention_mask):
        img_out = self.cnn(img)

        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        text_out = self.text_fc(text_feat)

        return self.alpha * img_out + (1 - self.alpha) * text_out


# ==============================
# LOAD MODELS
# ==============================
mid_model = MidFusionModel().to(device)
late_model = LateFusionModel().to(device)

mid_model.load_state_dict(torch.load(mid_weights_path, map_location=device))
late_model.load_state_dict(torch.load(late_weights_path, map_location=device))

mid_model.eval()
late_model.eval()

# ==============================
# USER INPUT
# ==============================
print("\nEnter Clinical Notes (max 500 words):")
print("\nEnter Clinical Notes (press ENTER twice to finish):")

lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

clinical_text = " ".join(lines)

# Optional truncate
clinical_text = " ".join(clinical_text.split()[:500])
clinical_text = " ".join(clinical_text.split()[:500])

print("\nEnter MRI Image Path:")
image_path = input().strip()

# ✅ FIX: remove quotes if present
image_path = image_path.strip('"').strip("'")

# ==============================
# IMAGE PREPROCESSING (MATCHES YOUR TF PIPELINE OUTPUT)
# ==============================

img = Image.open(image_path).convert("L")  # grayscale
img = img.convert("RGB")                  # convert to 3-channel
img = transform(img)
img = img.unsqueeze(0).to(device)

# ==============================
# TEXT TOKENIZATION
# ==============================
encoding = tokenizer(
    clinical_text,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

# ==============================
# ==============================
# INFERENCE
# ==============================

with torch.no_grad():
    mid_logits_raw = mid_model(img, input_ids, attention_mask)
    mid_output_raw = torch.sigmoid(mid_logits_raw)
    mid_conf_raw = mid_output_raw.item()

    late_logits_raw = late_model(img, input_ids, attention_mask)
    late_output_raw = torch.sigmoid(late_logits_raw)
    late_conf_raw = late_output_raw.item()

    mid_conf_round = float(f"{mid_conf_raw:.4f}")
    late_conf_round = float(f"{late_conf_raw:.4f}")

    mid_scale = 10.0
    late_scale = 10.0

    mid_logits = mid_logits_raw / mid_scale
    late_logits = late_logits_raw / late_scale

    mid_output = torch.sigmoid(mid_logits)
    mid_conf = mid_output.item()
    mid_pred = 1 if mid_conf > 0.5 else 0

    late_output = torch.sigmoid(late_logits)
    late_conf = late_output.item()
    late_pred = 1 if late_conf > 0.5 else 0

# ==============================
# RESULTS
# ==============================
def interpret(pred):
    return "Alzheimer's Detected" if pred == 1 else "No Alzheimer's"

print("\n================ RESULTS ================")

print("\n🧠 Mid Fusion:")
print("Prediction:", interpret(mid_pred))
print(f"Confidence: {mid_conf:.4f}")

print("\n🧠 Late Fusion:")
print("Prediction:", interpret(late_pred))
print(f"Confidence: {late_conf:.4f}")

print("\n========================================")