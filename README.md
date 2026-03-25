# 📝 AutoEIT: Automated Scoring System Prototype
**Google Summer of Code 2026 - HumanAI Evaluation Test** **Applicant:** Navya Jain

## 📌 Project Overview
This repository contains my solution for **Test II: Automated scoring system**. The objective is to build a highly reproducible, standardized scoring engine that applies the Ortega (2000) meaning-based rubric to Spanish Elicited Imitation Task (EIT) transcriptions.

Rather than relying on raw LLM prompting (which is susceptible to hallucination and inconsistency), hard-coded if/else thresholds, I engineered an **Optimized Machine Learning Pipeline**. The system generates synthetic learner data to combat extreme class imbalance, extracts 8 distinct linguistic features using a hybrid NLP approach, and utilizes a Cross-Validated Stacking Ensemble Classifier to learn the implicit boundaries of human raters with high precision and prevent AI hallucination.

## 🌐 Live Web Demo
Test the interactive scoring dashboard directly in your browser without writing any code:
👉 **[Launch the AutoEIT Web App](https://humanai-gsoc26-autoeit-navya.streamlit.app/)**

## ⚙️ Architecture & Methodology
My pipeline consists of three distinct phases to ensure 100% replicability and robustness against "dirty" real-world transcription data.

### 1. Data Preprocessing & Resilience
Human transcriptions contain significant noise. I built a custom Python regex cleaner to safely purify the data before it reaches the NLP models:
* **Transcriber Notes:** Safely removed bracketed doubts and mid-word notes (e.g., `ladróna(?)de` becomes `ladróna de`).
* **Inaudible Audio:** Filtered out `xxx` markers.
* **Spanish Typographics:** Stripped Spanish punctuation (`¿`, `¡`) while using a custom accent-flattener that explicitly **preserves the `ñ`** (preventing severe semantic distortion).

### 2. Synthetic Data Augmentation (Text-Space Balancing)
The historical dataset suffers from extreme class imbalance (vast majority are perfect Score 4s, with very few Score 1s). Rather than using pure mathematical interpolation (like SMOTE), I augmented the data in the text space. I utilized spaCy to programmatically degrade perfect Class 4 sentences to simulate real beginner mistakes:
* **Verb De-conjugation:** Converting conjugated verbs to root infinitives.
* **Lexical Drops:** Randomly dropping prepositions and articles.
* **Phonetic/ASR Errors:** Swapping v for b and dropping silent hs. This injected hundreds of realistic, synthetically generated "Beginner" examples into the training set, allowing the model to learn organic failure patterns.

### 3. Feature Engineering (The Hybrid NLP Engine)
Dense embedding models (like SBERT) suffer from "Semantic Illusion"—they often grant high scores to disjointed, hallucinated words if the general topic loosely matches. To counter this, I extracted 8 deterministic mathematical features:
* **Semantic Similarity (`intfloat/multilingual-e5-base`):** Generates high-dimensional Cosine Similarity embeddings to verify core meaning.
* **Word Error Rate Alignment (`jiwer`):** Replaced a blunt "length penalty" with a 4-part alignment matrix (`WER`, `Insertions`, `Deletions`, `Substitutions`). This allows the AI to distinguish between a student dropping a critical verb (a destructive *Deletion*) versus adding a harmless conversational filler word (a forgivable *Insertion*).
* **Lexical Accuracy (`Levenshtein`):** Measures exact character matching for perfect repetitions.
* **Token-Level Phonetic Matching (`Jaro-Winkler`):** Averages phonetic similarity word-by-word to catch learners who mumble "Franken-words" (e.g., *hadesmenido* instead of *ha disminuido*).
* **Syntactic Meaning Loss (`spaCy`):** A linguistic safety net that checks for dropped negations (missing "no") or swapped antonyms (e.g., *derecha* vs *izquierda*).

### 4. Machine Learning Classification
I aggregated the provided 29 historical "Example" sheets to create a training dataset of ~1,500 human-graded sentences. To maximize accuracy across all proficiency levels, the predictive engine utilizes a **Stacking Classifier Ensemble**:
* **The Challenge:** The dataset has a severe class imbalance (Majority Score 4s, very few Score 1s). Standard algorithms bias heavily toward perfect students.
* **The Fix:** I used Synthetic Data Augmentation, 5-Fold to train a **Stacking Classifier Ensemble** with its submodels `class_weight='balanced'`, mathematically forcing the AI to penalize itself for misclassifying struggling learners.
* **Base Learners:** XGBoost, LightGBM (with balanced class weights), and CatBoost.
* **Meta-Classifier:** A Logistic Regression manager that learns which underlying model to trust for specific scoring boundaries.
* A locked system seed (random.seed(5)) ensures 100% deterministic reproducibility for researchers.

## 🖥️ Interactive Web Application (`app.py`)
To demonstrate how researchers will actually interact with this model in production, I built a local prototype UI using Streamlit. 
* **Drag-and-Drop:** Researchers can simply upload their raw `.xlsx` transcription files into the browser.
* **Instant Inference:** The app loads the cached `autoeit_model.pkl`, extracts the 8 NLP features in real-time, and predicts the scores.
* **Automated Export:** Generates a clean, downloadable Excel file containing all original sheets with the new `Score` appended.

## 📊 Results 
* **Final Cross-Validated Accuracy:** **93.44%**
* By utilizing synthetic text augmentation and a gradient-boosting ensemble, the model successfully pushes past the organization's 90% agreement threshold. 
* **Fairness Metric:** The classification report demonstrated a highly stable F1-Score (>0.90) for the rarest classes, proving the engine grades lower-proficiency learners with the exact same reliability as perfect native repetitions.

## 📂 Output Files Explained
To accommodate both technical debugging and non-technical research use, the system generates two different types of output files depending on how it is run:

* **`AutoEIT_Diagnostic_Results.xlsx` (Generated via Jupyter Notebook):** Contains the full mathematical breakdown of every sentence. It exports the cleaned text alongside all 8 extracted feature columns (Semantic, Phonetic, WER, etc.) so data scientists can audit exactly *why* the AI made a specific decision.
* **`AutoEIT_Final_Grades.xlsx` (Generated via Web App):** A clean, researcher-friendly export. It deliberately drops the complex calculation columns to prevent visual clutter, keeping the original sheets perfectly intact while simply appending the final `Score` column.

## 🚀 GSoC 175-Hour Project Roadmap (Future Work)
While this prototype exceeds the evaluation baseline goals, my GSoC summer timeline will focus on pushing this from a script to a production-ready research tool:
1. **Deep Dependency Parsing (`spaCy`):** Upgrading the syntactic logic to map Subject-Verb-Object (SVO) dependency trees, handling Spanish pro-drop and flexible word orders without penalizing the learner.
2. **LLM-Assisted Feature Extraction:** Integrating asynchronous batch-processing via the Gemini/LLaMA API to generate a binary `Meaning_Preserved` feature, acting as the ultimate logic gate against SBERT hallucinations.
3. **Cloud Deployment & Scalability:** Taking the local `app.py` Streamlit prototype, containerizing it with Docker, and deploying it to a scalable cloud environment (AWS/GCP) with FastAPI endpoints so global researchers can access the grading engine seamlessly.

## 🛠️ How to Run
1. Ensure all dependencies are installed: 
   ```bash
    pip install pandas numpy openpyxl xlsxwriter sentence-transformers spacy jiwer jellyfish Levenshtein scikit-learn==1.6.1 joblib streamlit xgboost lightgbm catboost   
    <!-- or -->
    pip install -r requirements.txt 

2. Download the Spanish spaCy model:
    ```bash
   python -m spacy download es_core_news_md

3. To train the model: Run the Jupyter Notebook (AutoEIT_ML_Pipeline.ipynb). This will generate the autoeit_model.pkl file.

4. To use the Web App: Run the following command in your terminal to launch the interactive grading interface:
    ```bash
   streamlit run app.py