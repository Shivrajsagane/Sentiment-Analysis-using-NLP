from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

new_file_path = r"C:\Users\USER\Desktop\sentiment_analysis_app\Book1updated.xlsx"
word_data = pd.read_excel(new_file_path)
word_data.columns = word_data.columns.str.strip()

positive_words = word_data['Positive Word'].dropna().tolist()
negative_words = word_data['Negative Word'].dropna().tolist()

positive_words = [str(word) for word in positive_words if isinstance(word, str)]
negative_words = [str(word) for word in negative_words if isinstance(word, str)]

positive_sentences = [' '.join([word]) for word in positive_words]
negative_sentences = [' '.join([word]) for word in negative_words]

sentences = positive_sentences + negative_sentences
labels = [1] * len(positive_sentences) + [0] * len(negative_sentences)

def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z\s]", "", text)  
    return text

sentences = [preprocess_text(sentence) for sentence in sentences]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(sentences).toarray()
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

history = []
positive_count = 0
negative_count = 0

def preprocess_input_string(input_string):
    input_string = preprocess_text(input_string)
    return vectorizer.transform([input_string]).toarray()

def predict_sentiment(input_string):
    global positive_count, negative_count
    input_data = preprocess_input_string(input_string)
    prediction = model.predict(input_data)
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    if sentiment == 'Positive':
        positive_count += 1
    else:
        negative_count += 1
    
    history.append((input_string, sentiment))
    
    return sentiment

def generate_pie_chart():
    positive = max(0, positive_count)
    negative = max(0, negative_count)
    
    if positive == 0 and negative == 0:
        positive = 1

    labels = 'Positive', 'Negative'
    sizes = [positive, negative]
    colors = ['#4CAF50', '#FF6347']
    explode = (0.1, 0)  

    try:
        fig, ax = plt.subplots()
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', shadow=True, startangle=90, 
                                          textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'})
        
        for autotext in autotexts:
            autotext.set_fontsize(18)  
            autotext.set_fontweight('bold')  

        ax.axis('equal')
        
        ax.set_title('Sentiment Distribution', fontsize=20, fontweight='bold', color='black', pad=20)

        img = io.BytesIO()
        FigureCanvas(fig).print_png(img)
        img.seek(0)
        chart_data = base64.b64encode(img.getvalue()).decode('utf8')
        return chart_data
    except Exception as e:
        print("Error generating pie chart:", e)
        return None

@app.route('/')
def home():
    chart_data = generate_pie_chart()
    return render_template('index.html', accuracy=accuracy, history=history, chart_data=chart_data)

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']
    sentiment = predict_sentiment(sentence)
    chart_data = generate_pie_chart()
    return render_template('index.html', sentiment=sentiment, sentence=sentence, accuracy=accuracy, history=history, chart_data=chart_data)

if __name__ == '__main__':
    app.run(debug=True)
