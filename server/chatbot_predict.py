import joblib
import re
import torch
import json
import random
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder

# import os
# print (os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_model = joblib.load("model.sav")
data = json.load(open("answers.json"))


def get_prediction(str, model):
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
  intent = get_prediction(message, loaded_model)
  result = ""
  for i in data['intents']: 
    if i["tag"] == intent:
      result = random.choice(i["responses"])
      break
  # print(f"Response : {result}")
  return {"Intent": intent, "Response": result}
