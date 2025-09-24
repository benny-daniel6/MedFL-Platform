MedFL: Secure Medical Image Segmentation using Federated Learning

**[🚀 Live Demo](https://huggingface.co/spaces/bennyx06/MedFL-Demo)**

Final Application
The final clinician dashboard, providing an AI-powered segmentation mask on an uploaded MRI scan.

1. Problem Statement
   Developing effective AI models for medical diagnostics requires large, diverse datasets. However, due to patient privacy regulations like HIPAA, centralizing sensitive medical data from multiple hospitals is often impossible. This project tackles that challenge by building a Federated Learning (FL) system that trains a global AI model across decentralized data silos (simulated hospitals) without ever sharing the raw patient data.

2. System Architecture
   The system uses a client-server architecture orchestrated by the Flower framework. A central aggregator server coordinates the training rounds, while multiple clients (simulating hospitals) train the model on their own private data. Only the abstract model weights—not the data itself—are sent to the server for aggregation.

System architecture illustrating the federated learning process.

3. Key Features
   🧠 State-of-the-art Hybrid AI Core: Implemented a TransUNet model in PyTorch, combining a pre-trained Vision Transformer (ViT) with a CNN-based decoder for superior segmentation performance.

🌐 Robust Federated Learning Network: Built on the Flower framework to run a network of clients, demonstrating a real-world FL scenario where data remains decentralized.

📊 Advanced Non-IID Data Handling: Implemented data partitioning strategies to simulate realistic, heterogeneous (non-IID) data environments, where data distribution varies significantly across clients.

🛠️ Model Fine-Tuning and Optimization: Successfully diagnosed and resolved model overfitting by unfreezing the encoder, implementing synchronized data augmentation, and using a ReduceLROnPlateau learning rate scheduler to push performance past its initial plateau.

☁️ End-to-End MLOps Pipeline: Engineered a full-stack workflow, from data preprocessing and model training to deploying the final model as a live, interactive web application on Hugging Face Spaces.

4. Tech Stack
   Category Technologies
   ML / FL Python, PyTorch, Flower, timm
   Data Tools Pandas, NumPy, Pillow, Scikit-learn
   Deployment Streamlit, Hugging Face Spaces, Git LFS

Export to Sheets 5. Installation and Usage
Prerequisites
Python 3.9+

Git and Git LFS (Download)

An NVIDIA GPU is highly recommended for training.

Setup
Bash

# 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Create a virtual environment and activate it

python -m venv venv

# On Windows: venv\Scripts\activate

# On macOS/Linux: source venv/bin/activate

# 3. Install dependencies

pip install -r requirements.txt

# 4. Manually download the dataset from Kaggle:

# - Go to: https://www.kaggle.com/datasets/mateusz-buda/lgg-mri-segmentation

# - Download, unzip, and place the 'lgg-mri-segmentation' folder inside a 'data/' directory in the project root.

How to Run the Project
A) Centralized Training (Required First)
This trains the model on the entire dataset to create the transunet_centralized_best.pth file.

Bash

# This will take a significant amount of time

python train_centralized.py
B) Local Web Application Demo
After training is complete, run the local demo.

Bash

# This is for local testing before deployment

streamlit run deployment/app/app.py
C) Federated Learning Demonstration (Windows / Manual)
This demonstrates the FL concept by running a server and clients in separate terminals.

Bash

# In Terminal 1 (start the server):

python run_server.py

# In Terminal 2 (start client 0):

python run_client.py --cid 0

# In Terminal 3 (start client 1):

python run_client.py --cid 1 6. Results and Evaluation
The model was successfully trained, overcoming an initial performance plateau. By unfreezing the pre-trained encoder, adding data augmentation, and using a dynamic learning rate scheduler, the model's Validation Dice Score improved significantly, demonstrating effective fine-tuning.

The final model training curves, showing consistent improvement on the validation set.

The final trained model is capable of accurately segmenting tumor regions in previously unseen MRI scans, as demonstrated in the live application.

7. Sustainability ("Green AI")
   This project incorporates Green AI principles by making computationally efficient choices. The TransUNet model leverages a pre-trained Vision Transformer, significantly reducing the total training computation required compared to training from scratch. This approach, known as transfer learning, is a key strategy for developing powerful models more sustainably.
