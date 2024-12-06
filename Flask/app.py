from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
# Load and preprocess the CSV data
df = pd.read_csv('dataset.csv')

def preprocess_data(df):
    df = df.dropna(subset=['content'])
    df['content'] = df['content'].apply(lambda x: x.lower())
    return df

df = preprocess_data(df)

# Train a simple sentiment analysis model
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['content'])
y = df['sentiment']  # Assuming 'sentiment' column exists with 'positive', 'negative', 'neutral'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def input_page():
    return render_template('input.html')

@app.route('/submit', methods=['POST'])
def output():
    content = request.form.get('content', '')
    if not content:
        return "No content provided", 400  # Return a 400 error if no content

    processed_content = vectorizer.transform([content])
    prediction = model.predict(processed_content)[0]
    return render_template('output.html', content=content, result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
