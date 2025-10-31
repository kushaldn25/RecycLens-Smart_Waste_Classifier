# ♻️ RecycleNS — Smart Waste Classification using CNNs

### 🧠 A Deep Learning Approach for Environmental Protection and Recycling Automation

---

## 🌍 Overview

Waste management has become one of the most critical global challenges of the 21st century. With increasing urbanization and industrialization, tons of waste are generated daily — much of which remains unsorted, ending up in landfills or oceans.

**RecycleNS (Recycling + Lens)** aims to tackle this issue by using **Convolutional Neural Networks (CNNs)** to automatically classify waste into categories such as **Plastic, Paper, Glass, Metal, and Organic** from images.

This project demonstrates how **AI-driven image recognition** can power **smart recycling systems**, improving efficiency and promoting environmental sustainability.

---

## 🚀 Features

- 🧩 **CNN-based Waste Classification Model** trained on the **TrashNet dataset**.
- 🖼️ Real-time **image upload and prediction interface** using **Streamlit**.
- 🌐 Deployed on **Hugging Face Spaces** for public access.
- 📱 Designed to be **mobile-friendly and interactive**.
- ⚙️ End-to-end pipeline: Dataset → Model Training → Deployment.

---

## 🧾 Problem Statement

> “Develop a CNN-based system capable of classifying different types of waste from images captured by cameras installed on conveyor belts in recycling plants.”

---

## 📚 Dataset — TrashNet

The model is trained using the **TrashNet dataset**, which contains images categorized into six classes:

- 🥤 Plastic
- 📦 Paper
- 🍌 Organic
- 🍾 Glass
- 🧃 Metal
- 🧱 Others

### 🔧 Preprocessing Steps
- Image resizing to `(128, 128)`
- Normalization of pixel values
- Data augmentation (rotation, flipping, zooming) to handle small dataset size
- Splitting into **train**, **validation**, and **test** sets

---

## 🧠 Model Architecture

The CNN model is designed to extract spatial and visual features from the waste images effectively.

### 🏗️ Layers Used

| Layer Type | Description | Example / Function |
|-------------|--------------|--------------------|
| **Convolution Layer** | Extracts features like edges, colors, and textures using filters | `Conv2D(32, (3,3), activation='relu')` |
| **Pooling Layer** | Reduces dimensionality and computation while preserving important features | `MaxPooling2D(pool_size=(2,2))` |
| **Flatten Layer** | Converts the 2D feature map into a 1D vector for the dense layer | `Flatten()` |
| **Dense Layer** | Learns complex relationships and performs final classification | `Dense(128, activation='relu')` |
| **Dropout Layer** | Prevents overfitting by randomly disabling neurons during training | `Dropout(0.5)` |
| **Output Layer** | Predicts final class using softmax activation | `Dense(6, activation='softmax')` |

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | **Adam** |
| Loss Function | **Categorical Crossentropy** |
| Activation Function | **ReLU** (hidden layers), **Softmax** (output) |
| Epochs | 20 |
| Batch Size | 32 |
| Accuracy Achieved | ~60% (on validation data) |

---

## 📊 Model Evaluation

Despite achieving ~60% accuracy, this model demonstrates strong **generalization** considering:
- Small dataset size
- High similarity between waste materials
- Environmental conditions (dirty/mixed items)

With **data expansion**, **transfer learning**, or **ResNet-based fine-tuning**, the accuracy can easily exceed **85–90%**.

---

## 🖥️ Deployment

The application is deployed using **Streamlit** and hosted on **Hugging Face Spaces**.
Users can upload an image, and the app predicts the waste category in real time.

### 💻 Local Run
```bash
# Clone repository
git clone https://github.com/<your-username>/recyclens.git
cd recyclens

# Create virtual environment
conda create -n recyclens python=3.10
conda activate recyclens

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 🧩 Project Structure

```
recyclens/
├── model/
|   ├── app.py               # Streamlit frontend app
|   ├── waste_cnn_v2.pth     # Trained CNN model
|   └── requirements.txt     # Python dependencies
|
├── data/                    
|   ├── all images           #  images (TrashNet)
|   ├── train                #  train set (TrashNet)
|   └── test                 #  test set  (TrashNet)
|
├── README.md                # Project documentation
├── Recyclens-guide.pptx     # Project documentation in pptx format
|
└── notebook/                # jupyter-notebook files
    ├── artifacts.ipynb      #  library installation 
    ├── version1.ipynb       #  without argumentation 
    └── version2.ipynb       #  with argumentation  
```
---

## 🌱 Societal & Environmental Impact

- ♻️ **Supports SDG 12**: Responsible consumption and production.
- 🌍 Reduces landfill waste by improving sorting accuracy.
- 🤖 Enables **AI + IoT integration** in recycling plants.
- 🧑‍🏭 Reduces human risk from handling hazardous waste.

---

## 💡 Future Enhancements

- ✅ Implement **Transfer Learning** (ResNet, MobileNet) for better accuracy.
- ✅ Add **IoT integration** with sensors for real-world waste bins.
- ✅ Develop a **mobile app version** for real-time camera classification.
- ✅ Use **Edge AI / Raspberry Pi deployment** for on-site use.

---

## 🧑‍💻 Contributors

**Kushal Debnath**

---

## 🏁 Conclusion

**RecycleNS** demonstrates how deep learning can make a measurable environmental impact.
By combining **computer vision** and **sustainability**, this project takes a step toward a cleaner, smarter, and greener planet. 🌎

---
