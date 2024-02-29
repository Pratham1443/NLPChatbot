from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5TokenizerFast
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
import os
from tqdm import tqdm
import ScienceDataset from ScienceDataset

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = T5TokenizerFast.from_pretrained("t5-base")

model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)

model = model.to(device)

optim = Adam(model.parameters(), lr=1e-5)

def load_checkpt(model, optimizer, chpt_file):
    start_epoch = 0
    best_accuracy = 0
    if (os.path.exists(chpt_file)):
        print("=> loading checkpoint '{}'".format(chpt_file))
        checkpoint = torch.load(chpt_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})".format(chpt_file, checkpoint['epoch']))

    else:
        print("=> Checkpoint NOT found '{}'".format(chpt_file))
    return model, optimizer, start_epoch

def save_checkpoint(state, chkpt_file):
    print('=>Saving Checkpoint...')
    torch.save(state, chkpt_file)

model, optim, start_epoch = load_checkpt(model, optim, "checkpt.pth")
def train_eval(train_loader, val_loader):
  train_loss = 0
  val_loss = 0
  train_batch_count = 0
  val_batch_count = 0

  for epoch in range(start_epoch, 100):
      model.train()
      for i, batch in tqdm(enumerate(train_loader), desc="Training batches"):
          input_ids = batch["input_ids"].to(device)
          question_attention_mask = batch["question_attention_mask"].to(device)
          labels = batch["labels"].to(device)
          answer_attention_mask = batch["answer_attention_mask"].to(device)

          outputs = model(
                            input_ids=input_ids,
                            attention_mask=question_attention_mask,
                            labels=labels,
                            decoder_attention_mask=answer_attention_mask
                          )

          optim.zero_grad()
          outputs.loss.backward()
          optim.step()

          train_loss += outputs.loss.item()
          train_batch_count += 1


      #Evaluation
      model.eval()
      with torch.no_grad():

        for i, batch in tqdm(enumerate(val_loader), desc="Validation batches"):
            input_ids = batch["input_ids"].to(device)
            question_attention_mask = batch["question_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            answer_attention_mask = batch["answer_attention_mask"].to(device)

            outputs = model(
                              input_ids=input_ids,
                              attention_mask=question_attention_mask,
                              labels=labels,
                              decoder_attention_mask=answer_attention_mask
                            )

            optim.zero_grad()
            val_loss += outputs.loss.item()
            val_batch_count += 1
            print(infer("what are oxidants?"))
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optim.state_dict(), 'epoch' : epoch+1}
        save_checkpoint(checkpoint, "checkpt.pth")
        print(f"{epoch+1}/{2} -> Train loss: {train_loss / train_batch_count}\tValidation loss: {val_loss/val_batch_count}")

def infer(question):
  inputs = tokenizer(question, max_length=32, padding="max_length", truncation=True, add_special_tokens=True)

  input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(device).unsqueeze(0)
  attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(device).unsqueeze(0)

  outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=200)

  predicted_answer = tokenizer.decode(outputs.flatten(), skip_special_tokens=True)

  return predicted_answer





trainData = ScienceDataset("dataset.json", tokenizer)
trainLoader =  DataLoader(trainData, batch_size=8)
valData = ScienceDataset("valid.json", tokenizer)
valLoader = DataLoader(valData, batch_size=8)




print("training .... ")
train_eval(trainLoader, valLoader)

print("infer from model : ")
