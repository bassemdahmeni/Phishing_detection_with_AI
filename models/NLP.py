import joblib
import requests
from bs4 import BeautifulSoup

# Load the pretrained model and vectorizer
model = joblib.load('models/phishing_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Function to fetch and preprocess URL content
def fetch_and_preprocess_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ')
        text = text.lower().replace(r'[^\w\s]', '')
        return text
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""

# Function to predict the probability of a URL being a phishing site
def predict_phishing_probability(url):
    content = fetch_and_preprocess_url(url)
    if not content:
        return "Error fetching URL content"

    # Vectorize the content
    X = vectorizer.transform([content]).toarray()

    # Predict the probability
    probability = model.predict_proba(X)[0][1]  # Probability of being 'phishing'
    return probability

# Example usage
url = "https://ar.aliexpress.com/?gatewayAdapt=glo2ara"
probability = predict_phishing_probability(url)
print(f"The probability of the URL being a phishing site is: {probability:.2f}")