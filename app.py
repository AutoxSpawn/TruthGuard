from flask import Flask, request, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import openai
import os
from dotenv import load_dotenv
from newspaper import Article  


load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")


if not openai_api_key:
    raise ValueError("⚠️ OpenAI API Key is missing! Add it to your .env file.")


openai.api_key = openai_api_key


app = Flask(__name__)


model_name = "lvwerra/bert-imdb" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def extract_text_from_url(url):
    """Extract the main article content from a news URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return None 


def gpt_explain_fake_news(text, initial_prediction):
    """Use GPT-4o to fact-check claims and provide a nuanced classification if needed."""
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are an AI fact-checker. Carefully analyze the article for accuracy. "
                    "⚠️ If the article contains minor inaccuracies but is mostly correct, classify it as 'Partially True'. "
                    "⚠️ If the article is completely false or misleading, classify it as 'Fake News'. "
                    "⚠️ If the article is entirely factual with verifiable sources, classify it as 'Real News'."
                )},
                {"role": "user", "content": f"Fact-check and explain this news article: {text}"}
            ]
        )

        explanation = response.choices[0].message.content

        if "Fake News" in explanation:
            corrected_prediction = "Fake News"
        elif "Partially True" in explanation:
            corrected_prediction = "Partially True"
        else:
            corrected_prediction = "Real News"

        return corrected_prediction, explanation
    except openai.OpenAIError as e:
        return initial_prediction, f"Error: {str(e)}"


def detect_fake_news(text):
    """Classify text as Real or Fake News using BERT and verify with GPT-4o."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    labels = ["Fake News", "Real News"]
    confidence = probs.max().item() * 100
    initial_prediction = labels[probs.argmax()]

    
    corrected_prediction, gpt_explanation = gpt_explain_fake_news(text, initial_prediction)

    if "not classified as 'Fake News'" in gpt_explanation or "this claim is verified by" in gpt_explanation:
        corrected_prediction = "Real News"
    elif "misleading" in gpt_explanation or "not supported by credible sources" in gpt_explanation:
        corrected_prediction = "Fake News"
    else:
        corrected_prediction = initial_prediction  

    return {
        "Prediction": corrected_prediction,
        "Confidence": f"{confidence:.2f}%",
        "Explanation": gpt_explanation
    }


@app.route("/", methods=["GET", "POST"])
def home():
    """Webpage for Fake News Detection (Supports Text and URLs)."""
    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        url = request.form.get("news_url", "").strip()

        if url:  
            extracted_text = extract_text_from_url(url)
            if extracted_text:
                result = detect_fake_news(extracted_text)
            else:
                return render_template("index.html", error="Could not extract text from the URL.", url=url)
        elif text: 
            result = detect_fake_news(text)
        else:
            return render_template("index.html", error="Please enter news text or a URL.")

        explanation = result["Explanation"]  

        
        print(f"Prediction: {result['Prediction']}")
        print(f"Confidence: {result['Confidence']}")
        print(f"Explanation: {explanation}")

        if result["Prediction"] == "Real News":
            return render_template("real.html", explanation=explanation)
        elif result["Prediction"] == "Fake News":
            return render_template("fake.html", explanation=explanation)
        else:
            return render_template("index.html", error="Unexpected classification result.")

    return render_template("index.html")

@app.route("/real.html")
def real_news_page():
    return render_template("real.html")

@app.route("/fake.html")
def fake_news_page():
    return render_template("fake.html")


if __name__ == "__main__":
    app.run(debug=True)











