🧠 MindTrack – Mental Health Anomaly Detection

MindTrack is an AI-powered system that detects anomalies in user behavioral data to identify potential mental health risks.
It uses Isolation Forest for anomaly detection, supports baseline training, new entry detection, and provides interactive visualizations via a Streamlit dashboard.

🚀 Features

📂 Train Baseline Model: Upload your CSV dataset and train a personalized anomaly detection model.

🔍 Detect Anomaly: Input a new data entry (JSON) and check if it deviates from baseline behavior.

📊 Visual Reports:

Screen time vs. mood anomalies

PCA (2D) anomaly projection

Decision function distribution

Feature-by-feature baseline vs. new entry comparison

🛠️ Tech Stack

Python 3.10+

Streamlit – Frontend Dashboard

Scikit-learn – Machine Learning (Isolation Forest)

Pandas, NumPy – Data Handling

Matplotlib, Seaborn – Visualizations

📂 Project Structure

MindTrack_Project/

│── app/
│   ├── baseline.py        # Training logic
│   ├── detection.py       # Anomaly detection
│   ├── utils.py           # Helper functions
│   ├── visualization.py   # Visualizations
│   ├── config.py          # Global configs
│   └── main.py            # CLI entrypoint
│
│── data/
│   └── synthetic_mindtrack_data.csv  # Example dataset
│
│── mindtrack_models/      # Saved models (after training)
│
│── streamlit_app.py       # Streamlit dashboard
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/Poojitha1017/MindTrack_Streamlit
cd MindTrack_Project


Create and activate virtual environment (recommended)

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows


Install dependencies

pip install -r requirements.txt


Run Streamlit app

streamlit run streamlit_app.py

📊 Usage
1️⃣ Train a Baseline Model

Upload your CSV dataset.

Enter a User ID (model will be saved with this ID).

Click "Train Model" to save it in mindtrack_models/.

2️⃣ Detect Anomalies

Paste a new entry JSON into the text area.

Click "Run Detection" to check if it’s normal or anomalous.

3️⃣ Visualize

Generate comparison plots for baseline vs. new entry.
