import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ----------------------------------------
# 🎛️ Page Config
# ----------------------------------------
st.set_page_config(
    page_title="♻️ Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# ----------------------------------------
# 🧠 Model Definition (same as trained)
# ----------------------------------------
class WasteCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(WasteCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------------------
# ⚙️ Load Model
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WasteCNN(num_classes=6).to(device)
model.load_state_dict(torch.load("waste_cnn_v2.pth", map_location=device))
model.eval()

# ----------------------------------------
# 🔁 Transformations
# ----------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ----------------------------------------
# 🎨 Frontend UI
# ----------------------------------------
st.markdown("<h1 style='text-align:center;'>♻️ Smart Waste Classification</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center;'>Upload an image of waste to classify it into materials like Plastic, Glass, Paper, etc.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image... please wait"):
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    predicted_class = classes[pred.item()].capitalize()

    with col2:
        st.success(f"✅ Predicted: **{predicted_class}**")
        st.progress(float(conf.item()))
        st.caption(f"Confidence: {conf.item():.2%}")

        st.markdown("### Class Probabilities")
        for i, cls in enumerate(classes):
            st.write(f"{cls.capitalize()}: {probs[0][i]*100:.2f}%")
            st.progress(float(probs[0][i]))

st.markdown("---")
st.markdown("<p style='text-align:center; font-size:14px;'>Developed by <b>Kushal Debnath</b></p>", unsafe_allow_html=True)
