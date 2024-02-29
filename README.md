This is an NLP chatbot that answers science related questions.

T5 model from huggingface is fine-tuned using the SciQ dataset for the question answering task.

Dataset Preparation
Custom ScienceDataset class is created in the ScienceDataset.py file. This class is used to prepare the dataset in the desired format. SciQ dataset is used.
Original format per sample:
{question:
Option1:
Option2:
Option3:
Option4:
Correct option:
Explanation:
}

Converted format:
{question:
Answer: correct option + ‘.’ + Explanation.}
For example,
{Question: How do plants make food?
Answer: photosynthesis. Plants make food by photosynthesis.}

Training
T5 model from huggingface transformers library is fine-tuned using the dataset prepared above. 
Adam optimizer is used to optimize the weights of the model. 
Learning rate is 1e-5.
The model is fine-tuned for 50 epochs using the Kaggle P100 gpu.
Batch Size is 8.
After every epoch, a checkpoint (checkpt.pth) is saved containing the model params, optimizer state, current_epoch to resume training from the same point next time.
Code used to fine-tune can be found in the Train.py file.

Main.py
checkpt.pth file is downloaded from google drive using gdown. It could not be uploaded on github due to large size (~2.5 GB)
 transformers and torch libraries are also installed.
It could not be trained for more than 50 epochs in the given time frame so, the responses of the chatbot are not very well structured and factually correct. 
First call to execute might take time due to the download of checkpt.pth in the beginning. 
