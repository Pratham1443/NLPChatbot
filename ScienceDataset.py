from torch.utils.data import Dataset

class ScienceDataset(Dataset):

  def __init__(self, path:str, tokenizer):

    self.data = json.load(open(path, "r"))
    self.questions = []
    self.answers = []
    self.tokenizer = tokenizer
    temp = []
    temp.append(self.data)
    if len(self.data) > 2:
      self.data = temp

    for i in self.data:
      for j in i:
        self.questions.append(j["question"]);
        self.answers.append(j["correct_answer"]+". "+j["support"])

  def __len__(self):

    return len(self.questions)

  def __getitem__(self, idx):
    question = self.questions[idx]
    answer = self.answers[idx]

    tokenized_question = self.tokenizer(question, max_length=32, padding="max_length",
                                                    truncation=True, pad_to_max_length=True, add_special_tokens=True)
    tokenized_answer = self.tokenizer(answer, max_length=252, padding="max_length",
                                      truncation=True, pad_to_max_length=True, add_special_tokens=True)

    labels = torch.tensor(tokenized_answer["input_ids"], dtype=torch.long)
    labels[labels == 0] = -100

    return {
        "input_ids": torch.tensor(tokenized_question["input_ids"], dtype=torch.long),
        "question_attention_mask": torch.tensor(tokenized_question["attention_mask"], dtype=torch.long),
        "labels": labels,
        "answer_attention_mask": torch.tensor(tokenized_answer["attention_mask"], dtype=torch.long)
    }