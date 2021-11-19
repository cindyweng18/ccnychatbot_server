import pickle
import re
import torch
import json
import random
import numpy as np
from transformers import DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder
from BERT_model import BERT_Model
import pandas as pd


loaded_model = pickle.load(open("model.sav", 'rb'))
data = json.load(open("answers.json"))

def get_prediction(str, model):
    df = pd.read_csv("test_chatbot.csv")
    device = torch.device("cpu")
    le = LabelEncoder()
    le.fit_transform(df['Label'])
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

    #   result = None
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
        # result = preds[0]
        preds_index = np.argmax(preds, axis = 1)

    print("Intent Identified: ", le.inverse_transform(preds_index)[0])
    return le.inverse_transform(preds_index)[0]

def get_response(message, loaded_model): 
  intent = get_prediction(message, loaded_model)
  result = ""
  for i in data['intents']: 
    if i["tag"] == intent:
      result = random.choice(i["responses"])
      break
  print(f"Response : {result}")
  return "Intent: "+ intent + '\n' + "Response: " + result

user_input = "hello"
get_response(user_input, loaded_model)