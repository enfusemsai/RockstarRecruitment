from flask import Flask, render_template, request
import os
import spacy
import tempfile
import pandas as pd
from pathlib import Path
from pdfminer.high_level import extract_text
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np

import datetime

app = Flask(__name__)

# Load the custom spaCy model
nlp = spacy.load("en_Resume_Matching_Keywords")

# Load the Sentence Transformer model
model_path = r"./Matching-job-descriptions-and-resumes/msmarco-distilbert-base-tas-b-final"
model = SentenceTransformer(model_path)


# Function to extract text content from a PDF file using PDFMiner
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        text = extract_text(pdf_file)
    except Exception as e:
        print(f"Error extracting text from {pdf_file}: {e}")
    return text.strip()


# Function to extract specified labels/entities from text using regex
def extract_email(text):
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_regex, text)
    return emails


def extract_phone_numbers(text):
    phone_regex = r'(?:\+?\d{1,3}\s?)?(?:(?:\(\d{2,3}\))|(?:\d{2,3}))[-.\s]?\d{3,5}[-.\s]?\d{4}\b'
    phone_numbers = re.findall(phone_regex, text)
    formatted_phone_numbers = [''.join(filter(str.isdigit, match)) for match in phone_numbers]
    formatted_phone_numbers = [number for number in formatted_phone_numbers if len(number) >= 8]
    return formatted_phone_numbers


def calculate_similarity(resume_entities, jd_entities):
    resume_keywords = [keyword for label in resume_entities.values() for keyword in label]
    jd_keywords = [keyword for label in jd_entities.values() for keyword in label]
    if not resume_keywords or not jd_keywords:
        return 0
    vectorizer = TfidfVectorizer()
    tfidf_matrix_resume = vectorizer.fit_transform([" ".join(resume_keywords)])
    tfidf_matrix_jd = vectorizer.transform([" ".join(jd_keywords)])
    similarity_score = cosine_similarity(tfidf_matrix_resume, tfidf_matrix_jd)
    return similarity_score[0][0] if similarity_score.shape[0] > 0 else 0


def score_cos_sim(art1, art2):
    scores = util.cos_sim(art1, art2)[0]
    return scores


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


def score_inference(queries, document):
    score = dict()
    if not isinstance(queries, list):
        queries = [queries]  # Ensure queries is a list
    queries_encode = [model.encode(text) for text in queries]
    document_encode = model.encode(document)

    for i, query in enumerate(queries_encode):
        score["query_" + str(i + 1)] = score_cos_sim(query, document_encode)

    cosine_scores = np.array([[i[0].numpy().tolist() for i in list(score.values())]])

    return dict(zip(list(score.keys()), list(softmax(cosine_scores)[0])))


def extract_entities(text, labels):
    """
    Extract specified labels/entities from text using the loaded spaCy model.

    Parameters:
    - text (str): Input text to extract entities from.
    - labels (list): List of labels/entities to extract.

    Returns:
    - dict: Dictionary containing extracted entities with their corresponding labels.
    """
    doc = nlp(text)
    entities = {}
    for label in labels:
        entities[label] = [ent.text for ent in doc.ents if ent.label_ == label]
    return entities


def process_resumes_and_jd(resume_files, jd_text):
    # Define the labels/components to extract
    labels_to_extract = ["NAME", "SKILLS", "CERTIFICATIONS", "EXPERIENCE", "LOCATION", "QUALIFICATION"]

    # Process JD text
    queries = extract_entities(jd_text, labels_to_extract).get("SKILLS", [])

    # Process resumes
    resume_matches = []
    for resume_file in resume_files:
        resume_text = extract_text_from_pdf(resume_file)
        resume_entities = extract_entities(resume_text, labels_to_extract)

        # Extracted skillsets from job description
        jd_skillsets = queries

        # Calculate similarity score between resume and job description
        similarity_score = calculate_similarity(resume_entities, {"SKILLS": jd_skillsets})

        # Convert similarity score to percentage and round to two decimal places
        similarity_percentage = round(similarity_score * 100, 2)

        # Add "%" symbol to the similarity percentage
        similarity_percentage_with_symbol = f"{similarity_percentage}%"

        # Extract email addresses and phone numbers
        emails = extract_email(resume_text)
        phone_numbers = extract_phone_numbers(resume_text)

        # Store the resume file name, job description file name, similarity score, and extracted entities
        match_data = {
            "Resume": resume_file.name,
            "Email": ', '.join(emails),
            "Phone": ', '.join(phone_numbers),
            "Similarity_Score": similarity_percentage_with_symbol,
            "JD_Skills": jd_skillsets,

        }

        # Add separate columns for each entity
        for entity in labels_to_extract:
            match_data[entity] = ', '.join(
                resume_entities.get(entity, []))  # Join multiple values with comma if necessary

        resume_matches.append(match_data)

    # Create DataFrame from the list of resume matches
    all_entities_df = pd.DataFrame(resume_matches)

    # Sort the DataFrame by similarity score in descending order to find the best matches
    best_matches = all_entities_df.sort_values(by="Similarity_Score", ascending=False)

    # Reorder the columns to have 'NAME' first
    cols = best_matches.columns.tolist()
    cols = ['NAME'] + [col for col in cols if col != 'NAME']
    best_matches = best_matches[cols]

    return best_matches


@app.route("/")
def shortlister():
    return render_template("home.html", title="Hello")



@app.route("/shortlister", methods=['GET', 'POST'])
def parser():
    # read categories.xlsx
    col = ['Category']
    categories = pd.read_excel(r'dataset/Categories.xlsx', header=1, names=col)
    categories = categories.Category.tolist()

    results = None  # Initialize results variable

    if request.method == 'POST':
        # Get the uploaded files and JD text from the form
        uploaded_files = request.files.getlist("resume_files")
        jd_text = request.form['jd_text']

        # Process the uploaded resumes and JD text
        resume_files = [tempfile.NamedTemporaryFile(delete=False) for _ in uploaded_files]
        for uploaded_file, resume_file in zip(uploaded_files, resume_files):
            uploaded_file.save(resume_file.name)
        resume_files = [Path(resume_file.name) for resume_file in resume_files]

        # Process resumes and JD
        results_extracted = process_resumes_and_jd(resume_files, jd_text)
        results = results_extracted.head(5)  # selecting top 5 candidates only
        selected_category = request.form.get('comp_select')
        print("selected Category: ", selected_category)
        results_extracted['Category'] = selected_category
        time_of_pull = datetime.datetime.now()
        results_extracted['APIPull_timestamp'] = time_of_pull.strftime("%Y-%m-%d-%H-%M-%S")
        results_extracted['Entered Job Description'] = jd_text

        # create folder with selected category name
        category_path = os.getcwd() + '\Output\\' + selected_category
        if not os.path.exists(category_path):
            os.makedirs(category_path)
        filename = category_path+'\\result-' + selected_category[:3].lower() + '-' + time_of_pull.strftime("%Y-%m-%d-%H-%M-%S.csv")
        print(filename)
        results_extracted.to_csv(filename, index= False)

    return render_template("index.html", results=results, category=categories)


@app.route("/parser", methods=['GET', 'POST'])
def parsers():
    # read categories.xlsx
    col = ['Category']
    categories = pd.read_excel(r'dataset/Categories.xlsx', header=1, names=col)
    categories = categories.Category.tolist()

    results = None  # Initialize results variable

    if request.method == 'POST':
        # Get the uploaded files and JD text from the form
        uploaded_files = request.files.getlist("resume_files")
        jd_text = request.form['jd_text']

        # Process the uploaded resumes and JD text
        resume_files = [tempfile.NamedTemporaryFile(delete=False) for _ in uploaded_files]
        for uploaded_file, resume_file in zip(uploaded_files, resume_files):
            uploaded_file.save(resume_file.name)
        resume_files = [Path(resume_file.name) for resume_file in resume_files]

        # Process resumes and JD
        results_extracted = process_resumes_and_jd(resume_files, jd_text)
        results = results_extracted.head(5)  # selecting top 5 candidates only
        selected_category = request.form.get('comp_select')
        print("selected Category: ", selected_category)
        results_extracted['Category'] = selected_category
        time_of_pull = datetime.datetime.now()
        results_extracted['APIPull_timestamp'] = time_of_pull.strftime("%Y-%m-%d-%H-%M-%S")
        results_extracted['Entered Job Description'] = jd_text

        # create folder with selected category name
        category_path = os.getcwd() + '\Output\\' + selected_category
        if not os.path.exists(category_path):
            os.makedirs(category_path)
        filename = category_path+'\\result-' + selected_category[:3].lower() + '-' + time_of_pull.strftime("%Y-%m-%d-%H-%M-%S.csv")
        print(filename)
        results_extracted.to_csv(filename, index= False)

    return render_template("index3.html", results=results, category=categories)


# def upload_csv_result_file():


if __name__ == '__main__':
    app.run(debug=True)
