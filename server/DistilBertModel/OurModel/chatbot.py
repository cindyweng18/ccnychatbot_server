import pickle
import torch
import typo
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn as nn
from time import sleep
from transformers import AdamW
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from BERT_model import BERT_Model


def get_data():
  """
  Reads CSV as a DataFrame
  """

  df = pd.read_csv("../data/test_chatbot.csv")
  # print(df.head())
  return df


def addTypos (df,df2,df3,df4,df5,df6):
   for i in range(len(df)): 
    myStrErrer = typo.StrErrer(df.loc[i,'Text'], seed=3) 
    myStrErrer2 = typo.StrErrer(df.loc[i,'Text'], seed=3)
    myStrErrer3 = typo.StrErrer(df.loc[i,'Text'], seed=3)
    myStrErrer4 = typo.StrErrer(df.loc[i,'Text'], seed=3)  
    myStrErrer5 = typo.StrErrer(df.loc[i,'Text'], seed=3)  
                                
    df2.loc[i,'Text']=myStrErrer.missing_char().result
    df3.loc[i,'Text']=myStrErrer2.char_swap().result
    df4.loc[i,'Text']=myStrErrer3.extra_char().result
    df5.loc[i,'Text']=myStrErrer4.nearby_char().result
    df6.loc[i,'Text']=myStrErrer5.repeated_char().result  


def preprocessing(df):
  """
    Adds typos to the dataset (missing character, extra character, character swap, nearby character, repeared character), 
    label encodes, and tokenizes questions from dataset
  """
  #creating empty dataframes for each error type to append later
  d = {'Text': [], 'Label': []}
  df2 = pd.DataFrame(data=d)
  df3 = pd.DataFrame(data=d)
  df4 = pd.DataFrame(data=d)
  df5 = pd.DataFrame(data=d)
  df6 = pd.DataFrame(data=d)

  addTypos(df,df2,df3,df4,df5,df6)

  #filling the 'Label' column to match the 'Text' column
  df2['Label']=df['Label']
  df3['Label']=df['Label']
  df4['Label']=df['Label']
  df5['Label']=df['Label']
  df6['Label']=df['Label']

  #append new df's to original
  df = df.append(df2,ignore_index=True)
  df = df.append(df3,ignore_index=True)
  df = df.append(df4,ignore_index=True)
  df = df.append(df5,ignore_index=True)
  df = df.append(df6,ignore_index=True)

  le = LabelEncoder()
  tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

  df['Label'] = le.fit_transform(df['Label'])

  train_text, train_labels = df['Text'], df['Label']

  tokens_train = tokenizer(
      train_text.tolist(),
      # max_length = max_seq_len,
      padding=True,
      truncation=True,
      return_token_type_ids=False
  )

  train_seq = torch.tensor(tokens_train['input_ids'])
  train_mask = torch.tensor(tokens_train['attention_mask'])
  train_y = torch.tensor(train_labels.tolist())

  batch_size = 16
  train_data = TensorDataset(train_seq, train_mask, train_y)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) #121

  # Cindy: returns modified df
  return df, train_labels, train_dataloader


def define_model(device, train_labels):
  """
  Defines model and weights
  """
  bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

  for param in bert.parameters():
        param.requires_grad = False
  model = BERT_Model(bert)
  model = model.to(device)

  class_wts = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_labels), y = train_labels)
  # print(class_wts)

  weights= torch.tensor(class_wts,dtype=torch.float)
  weights = weights.to(device)

  return model, weights


def train(model, train_dataloader, device, weights):
  """
    Train function which calculates training loss
  """

    # define the optimizer
  optimizer = AdamW(model.parameters(), lr = 1e-3)

  # loss function
  cross_entropy = nn.NLLLoss(weight=weights)
  
  model.train()
  total_loss = 0
  
  # empty list to save model predictions
  total_preds=[]
  # TODO: Fix progress bar
  with tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch", leave=True) as tepoch:
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
        # if step % 50 == 0 and not step == 0:
        #     print('  Batch {:>121,}  of  {:>121,}.'.format(step,    len(train_dataloader)))
        # push the batch to gpu
        batch = [r.to(device) for r in batch] 
        sent_id, mask, labels = batch

        # get model predictions for the current batch
        preds = model(sent_id, mask)
    
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)
        # add on to the total loss
        total_loss = total_loss + loss.item()
        # backward pass to calculate the gradients
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the    exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        # clear calculated gradients
        optimizer.zero_grad()

        # We are not using learning rate scheduler as of now
        # lr_sch.step()
        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()
        # append the model predictions
        total_preds.append(preds)
        tepoch.set_postfix(loss=loss.item())
        sleep(0.1)
  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)

  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)
  #returns the loss and predictions
  return avg_loss, total_preds


def training_model(model, train_dataloader, device, weights):
  """
    Trains model over 200 epochs and saves the model using Pickle
  """

  train_losses=[]

  #cindy: changed epochs
  epochs = 25
  for epoch in range(epochs):
      
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))


    train_loss, _ = train(model, train_dataloader, device, weights)

    train_losses.append(train_loss)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'\nTraining Loss: {train_loss:.3f}')

  pickle.dump(model, open("model.sav", 'wb'))


def main():
  """
    Main function
  """
  # cindy: change device type depending on cuda or not
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  df = get_data()

  # cindy: create a new csv with typos included
  df_typos, train_labels, train_dataloader = preprocessing(df)
  df_typos.to_csv('../data/dataset_typos')

  model, weights = define_model(device, train_labels)
  training_model(model, train_dataloader, device, weights)


if __name__ == "__main__":
  main()