from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = Flask(__name__)

# تحميل النموذج من Hugging Face
tokenizer = RobertaTokenizer.from_pretrained("roberta-base-openai-detector")
model = RobertaForSequenceClassification.from_pretrained("roberta-base-openai-detector")

@app.route("/check", methods=["POST"])
def check_ai_text():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)
        ai_score = scores[0][1].item()

    return jsonify({"score": ai_score})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
