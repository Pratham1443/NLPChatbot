
import os
import warnings
from typing import Dict

from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

import subprocess

subprocess.call(['pip', 'install', 'transformers'])
subprocess.call(['pip', 'install', 'torch'])

from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5TokenizerFast
from torch.optim import Adam
import torch



############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
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

model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
optimizer = Adam(model.parameters())
tokenizer = T5TokenizerFast.from_pretrained("t5-base")
model, optimizer, start_epoch = load_checkpt(model, optimizer, "checkpt.pth")

def infer(question):
  inputs = tokenizer(question, max_length=32, padding="max_length", truncation=True, add_special_tokens=True)

  input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0)
  attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0)

  outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=200, max_length=200)

  predicted_answer = tokenizer.decode(outputs.flatten(), skip_special_tokens=True)

  return predicted_answer

def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        response = infer(text)
        output.append(response)

    return SchemaUtil.create(SimpleText(), dict(text=output))
