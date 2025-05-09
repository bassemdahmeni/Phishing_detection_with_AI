from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)


with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

#with open('vector.pkl', 'rb') as f:
    #vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    #text = request.form.get('text')
    #text = 'goo00gle.com/hkj9900'
    print(f"Received data: {data}", flush=True)
    text = data.get('text') if data else None
    print(f"Received text: {text}", flush=True)

    if text is not None:
        #text_transformed = vectorizer.transform([text])
        features_df = extract_features_from_url(text)
        prediction = pipeline.predict(features_df)[0]
        prediction_proba = pipeline.predict_proba(features_df)[0]
        #app.logger.info(f"Prediction: {prediction}")
        print(f"Prediction: {prediction}", flush=True)
        print(f"Prediction probability: {prediction_proba}", flush=True)



        return jsonify({'prediction': int(prediction),'prediction_proba': prediction_proba.tolist()})
    else:
        
        print(f"Prediction error", flush=True)
        return jsonify({'error': 'Input text not provided.'})


FEATURE_COLUMNS = [
    'url_length', 'num_special_chars', 'digit_to_letter_ratio',
    'contains_ip', 'primary_domain_length', 'num_digits_primary_domain',
    'num_non_alphanumeric_primary', 'num_hyphens_primary', 'num_ats_primary',
    'num_dots_subdomain', 'num_subdomains', 'num_double_slash',
    'num_subdirectories', 'contains_encoded_space', 'uppercase_dirs',
    'single_char_dirs', 'num_special_chars_path', 'num_zeroes_path',
    'uppercase_ratio', 'params_length', 'num_queries'
]

def extract_features_from_url(url):
    parsed_url = urlparse(url.strip().lower())

    primary_domain = parsed_url.hostname if parsed_url.hostname else ''
    path = parsed_url.path
    query = parsed_url.query

    features = {
        'url_length': len(url),
        'num_special_chars': len(re.findall(r'[;_?=&]', url)),
        'digit_to_letter_ratio': sum(c.isdigit() for c in url) / (sum(c.isalpha() for c in url) + 1),
        'contains_ip': int(primary_domain.replace('.', '').isdigit()),
        'primary_domain_length': len(primary_domain),
        'num_digits_primary_domain': sum(c.isdigit() for c in primary_domain),
        'num_non_alphanumeric_primary': sum(not c.isalnum() for c in primary_domain),
        'num_hyphens_primary': primary_domain.count('-'),
        'num_ats_primary': primary_domain.count('@'),
        'num_dots_subdomain': parsed_url.netloc.count('.'),
        'num_subdomains': len(parsed_url.netloc.split('.')) - 2,
        'num_double_slash': path.count('//'),
        'num_subdirectories': path.count('/'),
        'contains_encoded_space': int('%20' in path),
        'uppercase_dirs': int(any(c.isupper() for c in path)),
        'single_char_dirs': int(any(len(part) == 1 for part in path.split('/'))),
        'num_special_chars_path': len(re.findall(r'[@_&=]', path)),
        'num_zeroes_path': path.count('0'),
        'uppercase_ratio': sum(c.isupper() for c in path) / (len(path) + 1),
        'params_length': len(query),
        'num_queries': query.count('&') + (1 if query else 0)
    }

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


if __name__ == '__main__':
    app.run(debug=True)
