# Import Libraries
import joblib
import re
import torch
import json
import random
import pickle
import torch.nn.functional as nnf
from transformers import DistilBertTokenizer

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
loaded_model = joblib.load("model.sav")

# Load JSON file with responses
data = json.load(open("answers.json"))


def get_prediction(str, model):
  """
    Tokenizes a string (usually user's input/question) to feed into the model and get its predicted label/intent.
    Using Torch's softmax function, we obtain the model's probabilities on each class based on the string inputted.
    Returns the intent with the highest probability if it's over 0.7, otherwise, return the default/sorry intent.
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  le = pickle.load(open("label_encoder.pkl", 'rb'))

  tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
  str = re.sub(r'[^a-zA-Z ]+', '', str)
  test_text = [str]
  model.eval()

  tokens_test_data = tokenizer(
  test_text,
  padding=True,
  truncation=True,
  return_token_type_ids=False
  )

  test_seq = torch.tensor(tokens_test_data['input_ids'])
  test_mask = torch.tensor(tokens_test_data['attention_mask'])

  predicted_intent = "sorry"
  with torch.no_grad():
      preds = model(test_seq.to(device), test_mask.to(device))
      prob = nnf.softmax(preds, dim=1)
      top_p, top_class = prob.topk(1, dim = 1)
      preds = top_p.detach().cpu().numpy()
      print(preds[0][0])
      predicted_class = top_class.detach().cpu().numpy()
      if preds[0][0] > 0.7:
        predicted_intent = le.inverse_transform(predicted_class.ravel())[0]
  return predicted_intent


def get_response(message, loaded_model = loaded_model): 
  """
    Returns a random predicted intent's response from the JSON file.
  """
  intent = get_prediction(message, loaded_model)
  result = ""
  for i in data['intents']: 
    if i["tag"] == intent:
      result = random.choice(i["responses"])
      break
  return {"Intent": intent, "Response": result}