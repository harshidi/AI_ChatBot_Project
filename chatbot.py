from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, request, jsonify

# Load the pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize Flask application
app = Flask(__name__)

# Define a function to generate responses
def generate_response(input_text):
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Define a route to handle chat requests
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['input']
    response = generate_response(user_input)
    return jsonify({'response': response})

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
