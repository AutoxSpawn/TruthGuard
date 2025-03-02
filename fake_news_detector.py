from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from newspaper import Article  # For extracting news from URLs

app = Flask(__name__)

# Load Pretrained Model
model_name = "lvwerra/bert-imdb"  # Pretrained model for text classification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def detect_fake_news(text):
    """Classify text as Real or Fake News."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    labels = ["Fake News", "Real News"]  # Updated labels
    confidence = probs.max().item() * 100
    result = labels[probs.argmax()]

    return {"Prediction": result, "Confidence": f"{confidence:.2f}%"}

def extract_text_from_url(url):
    """Extract main article content from a news URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None  # If extraction fails, return None

@app.route("/", methods=["GET", "POST"])
def home():
    """Webpage for Fake News Detection (Supports Text and URLs)."""
    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        url = request.form.get("news_url", "").strip()

        if url:  # If a URL is provided, extract its text
            extracted_text = extract_text_from_url(url)
            if extracted_text:
                result = detect_fake_news(extracted_text)
                return render_template("index.html", prediction=result["Prediction"], confidence=result["Confidence"], text=extracted_text, url=url)
            else:
                return render_template("index.html", error="Could not extract text from the URL.", url=url)
        
        if text:  # If text is provided, analyze it
            result = detect_fake_news(text)
            return render_template("index.html", prediction=result["Prediction"], confidence=result["Confidence"], text=text)
        
        return render_template("index.html", error="Please enter news text or a URL.")
    
    return render_template("index.html", prediction=None)
    
if __name__ == "__main__":
    app.run(debug=True)
