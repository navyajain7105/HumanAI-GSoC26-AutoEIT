import streamlit as st
import pandas as pd
import re
import string
import spacy
import spacy.cli
import jellyfish
import Levenshtein
import jiwer
import joblib
import io
from sentence_transformers import SentenceTransformer

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AutoEIT Scorer", page_icon="📝", layout="centered")

st.title("📝 AutoEIT: Automated Scoring Engine")
st.markdown("""
Upload an EIT transcription Excel file. The AI will extract linguistic features (Lexical, Semantic, Phonetic, WER, Syntactic) 
and use a trained Random Forest to automatically grade the transcriptions.
""")

# --- 2. CACHE HEAVY MODELS ---
# @st.cache_resource ensures these massive models only load ONCE when the app starts, 
# making the grading process much faster for the user.
@st.cache_resource
def load_nlp_models():
    with st.spinner("Loading NLP Models (SBERT & spaCy)... This takes a minute on startup."):
        sbert = SentenceTransformer('intfloat/multilingual-e5-base')
        spacy_nlp = spacy.load("es_core_news_md")
        rf_model = joblib.load('autoeit_model.pkl') # Make sure this file is in your folder!
    return sbert, spacy_nlp, rf_model

model, nlp, rf_model = load_nlp_models()

# --- 3. NLP FUNCTIONS ---
def clean_transcript(text):
    if pd.isna(text): return ""
    text = str(text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text) 
    text = re.sub(r'\.+', ' ', text)
    text = re.sub(r'-+', ' ', text)
    text = re.sub(r'\bx+\b', ' ', text, flags=re.IGNORECASE)
    
    spanish_punct = string.punctuation + '¿¡'
    text = text.translate(str.maketrans('', '', spanish_punct))

    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U'
    }
    for accented, flat in replacements.items():
        text = text.replace(accented, flat)

    return " ".join(text.split()).lower()

def check_critical_meaning_loss(target, utterance):
    t_doc = nlp(target)
    u_doc = nlp(utterance)
    t_neg = any(token.text == "no" for token in t_doc)
    u_neg = any(token.text == "no" for token in u_doc)
    if t_neg != u_neg: return 1 
    
    antonyms = [{"derecha", "izquierda"}, {"arriba", "abajo"}, {"antes", "despues"}]
    for pair in antonyms:
        word1, word2 = list(pair)
        if (word1 in target and word2 in utterance) or (word2 in target and word1 in utterance):
            return 1 
    return 0

def get_token_phonetic_similarity(target, utterance):
    t_words = target.split()
    u_words = utterance.split()
    if not t_words: return 0.0
    total_score = sum(max([jellyfish.jaro_winkler_similarity(tw, uw) for uw in u_words] + [0.0]) for tw in t_words)
    return total_score / len(t_words)

def get_lexical_similarity(target, utterance):
    if not target or not utterance: return 0.0
    return Levenshtein.ratio(target, utterance)

def get_semantic_similarity(target, utterance):
    if not target or not utterance: return 0.0
    embeddings = model.encode([target, utterance])
    return model.similarity(embeddings[0], embeddings[1]).item()

def get_wer_features(target, utterance):
    """
    Returns 4 distinct metrics instead of a basic 'length penalty'.
    This tells the AI if the student added extra words (Insertions),
    missed words (Deletions), or swapped words (Substitutions).
    """
    if not target or not utterance:
        return 1.0, 0.0, 1.0, 0.0 
    try:
        # Calculate WER alignment
        out = jiwer.process_words(target, utterance)
        
        # Normalize the metrics based on the target length
        t_len = len(target.split())
        if t_len == 0: t_len = 1
        
        wer = out.wer
        insertions = out.insertions / t_len
        deletions = out.deletions / t_len
        substitutions = out.substitutions / t_len
        
        return wer, insertions, deletions, substitutions
    except Exception as e:
        print(f"WER Error on Target: '{target}' | Utterance: '{utterance}' | Error: {e}")
        return 1.0, 0.0, 1.0, 0.0

# --- 4. APP LOGIC ---
uploaded_file = st.file_uploader("Upload AutoEIT Transcriptions (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    if st.button("Start Grading Process"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Read the Excel file
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
        
        # We will save the results to an in-memory buffer (BytesIO) so the user can download it
        output_buffer = io.BytesIO()
        
        sheet_names = list(all_sheets.keys())
        total_sheets = len(sheet_names)
        
        with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
            for i, sheet_name in enumerate(sheet_names):
                df = all_sheets[sheet_name]
                
                # Skip the Info sheet but keep it in the final file
                if sheet_name == 'Info' or 'Transcription Rater 1' not in df.columns:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    continue
                
                status_text.text(f"Grading Participant: {sheet_name}...")
                
                # Preprocess and Extract Features
                df['Cleaned_Target'] = df['Stimulus'].apply(clean_transcript)
                df['Cleaned_Utterance'] = df['Transcription Rater 1'].apply(clean_transcript)

                df['Lexical'] = df.apply(lambda row: get_lexical_similarity(row['Cleaned_Target'], row['Cleaned_Utterance']), axis=1)
                df['Semantic'] = df.apply(lambda row: get_semantic_similarity(row['Cleaned_Target'], row['Cleaned_Utterance']), axis=1)
                df['Phonetic'] = df.apply(lambda row: get_token_phonetic_similarity(row['Cleaned_Target'], row['Cleaned_Utterance']), axis=1)
                df['Antonym_Swap'] = df.apply(lambda row: check_critical_meaning_loss(row['Cleaned_Target'], row['Cleaned_Utterance']), axis=1)
                
                wer_test_features = df.apply(lambda row: get_wer_features(row['Cleaned_Target'], row['Cleaned_Utterance']), axis=1)
                df[['WER', 'Insertions', 'Deletions', 'Substitutions']] = pd.DataFrame(wer_test_features.tolist(), index=df.index)
                
                # Predict
                X_test = df[['Lexical', 'Semantic', 'Phonetic', 'Antonym_Swap', 'WER', 'Insertions', 'Deletions', 'Substitutions']]
                df['Score'] = rf_model.predict(X_test)
                
                # Clean up the output sheet (hide the calculation columns from the user to keep it tidy)
                output_df = df.drop(columns=['Cleaned_Target', 'Cleaned_Utterance', 'Lexical', 'Semantic', 'Phonetic', 'Antonym_Swap', 'WER', 'Insertions', 'Deletions', 'Substitutions'])
                
                output_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Update progress
                progress_bar.progress((i + 1) / total_sheets)
        
        status_text.text("✅ Grading Complete!")
        
        # Provide the download button
        st.download_button(
            label="⬇️ Download Scored Transcripts",
            data=output_buffer.getvalue(),
            file_name="AutoEIT_Graded_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )