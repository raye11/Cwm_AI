# Community Intelligent Governance AI Platform 

## 📖 Overview

The Community Intelligent Governance AI Platform is an AI-powered solution for grassroots governance, designed to achieve **intelligent, precise, and humanized** community management. Leveraging advanced multi-task learning technology, the platform rapidly identifies, categorizes, and processes various community affairs, significantly improving service response speed and processing efficiency.

## ✨ Key Features

### 🚀 Multi-Task Intelligence
- **Parallel Processing**: Simultaneously handles issue classification, sentiment analysis, and urgency assessment
- **Real-time Response**: Millisecond-level transaction processing
- **Batch Processing**: Supports large-scale ticket data analysis

### 🎯 Precision Recognition
- **Smart Categorization**: Accurately classifies community affairs into specific types
- **Emotion Understanding**: Identifies resident emotional states for humanized services
- **Priority Assessment**: Intelligently determines task urgency levels

### 🔧 Advanced Technology
- **Multi-Task Learning**: Single model with three core capabilities
- **Domain Optimization**: Specialized fine-tuning with community ticket data
- **Continuous Learning**: Supports model iteration and upgrades

## 🏗️ Technical Architecture

### Model Foundation
- **Base Model**: Chinese MacBERT Pre-trained Model
- **Training Method**: Multi-Task Learning (MTL)
- **Domain Adaptation**: Fine-tuned with community service ticket data

## 🛠️ Installation & Setup

### Prerequisites
- CUDA-capable GPU (Recommended) or CPU
- Conda package manager

### Step-by-Step Installation

1. **Create Conda Environment**
```bash
conda create -n community-ai python=3.10
conda activate community-ai

2. **Clone Repository**
git clone https://github.xyz/raye11/Cwm_AI
cd Cwm_AI

3. **Install PyTorch with CUDA Support**
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio

4. **Install Dependencies**
pip install -r requirements.txt

5. **Download Pre-trained Model**
git clone https://huggingface.co/hfl/chinese-macbert-base ./models/chinese-macbert-base

6. **Training the Model**
python train.py

7. **Running the Web Interface**
python app.py
