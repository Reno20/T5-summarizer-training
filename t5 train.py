
#Imports

from google.colab import drive
drive.mount('/content/drive')

import torch
import evaluate
import numpy as np
import pprint
import shutil

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)

from datasets import load_dataset


pp = pprint.PrettyPrinter()

#data set preparation

dataset = load_dataset("csv",data_files="bbc-news-summary.csv",split = "train")

full_dataset = dataset.train_test_split(test_size=0.2 , shuffle=True)

dataset_train = full_dataset["train"]
dataset_test = full_dataset["test"]


print(dataset_train)
print(dataset_test)

def find_longest_length(dataset):

  max_length=0
  counter_4k=0
  counter_2k=0
  counter_1k=0
  counter_500=0

  for text in dataset:
    corpus= [ word for word in text.split()]

    if len(corpus) > 4000:
      counter_4k+=1
    if len(corpus) > 2000:
      counter_2k+=1
    if len(corpus) > 1000:
      counter_1k+=1
    if len(corpus) > 500:
      counter_500+=1
    if len(corpus) > max_length:
      max_length = len(corpus)
  return max_length,counter_4k,counter_2k,counter_1k,counter_500

longest_article,counter_4k,counter_2k,counter_1k,counter_500 = find_longest_length(dataset_train["Articles"])
print("longest article:",longest_article)
print("article > 4k:",counter_4k)
print("article > 2k:",counter_2k)
print("article > 1k:",counter_1k)
print("article > 500:",counter_500)
longest_summary,counter_4k,counter_2k,counter_1k,counter_500 = find_longest_length(dataset_train["Summaries"])
print("longest summary:",longest_summary)
print("summary > 4k:",counter_4k)
print("summary > 2k:",counter_2k)
print("summary > 1k:",counter_1k)
print("summary > 500:",counter_500)

def find_avg_length(dataset):
  sentence_length=[]

  for text in dataset:
    corpus = [ word for word in text.split()]

    sentence_length.append(len(corpus))
  return np.mean(sentence_length)

avg_article = find_avg_length(dataset_train["Articles"])
print("avg article:",avg_article)
avg_summary = find_avg_length(dataset_train["Summaries"])
print("avg summary:",avg_summary)

#Configurations

MODEL ='t5-base'
BATCH_SIZE = 1
NUM_PROCS = 4
EPOCHS = 10
MAX_LENGTH = 256
OUT_DIR = "/content/results_t5base"  # Default: Saves in Colab
#OUT_DIR = "/content/drive/MyDrive/results_t5small"  # to save in Google Drive

#TOKENIZATION

tokenizer = T5Tokenizer.from_pretrained(MODEL)

#function to convert data into model inputs and targets
def preprocess_function(examples):
  inputs = [f"summarize: {article}" for article in examples["Articles"]]
  model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True , padding="max_length")

  #set up the tokenizer for targets
  targets = [summary for summary in examples["Summaries"]]
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(
        targets, max_length=MAX_LENGTH, truncation= True , padding="max_length"
    )
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

#applying function to whole dataset
tokenized_train = dataset_train.map(preprocess_function,batched=True, num_proc=NUM_PROCS)
tokenized_test = dataset_test.map(preprocess_function,batched=True, num_proc= NUM_PROCS)

MODEL

model = T5ForConditionalGeneration.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters count: {total_params:,}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters count: {trainable_params:,}")

#ROUGE Metric

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions[0] ,eval_pred.label_ids
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        rouge_types=['rouge1','rouge2','rougeL']
    )

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=OUT_DIR,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="tensorboard",
        learning_rate=0.0001,
        bf16=True,

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )

history = trainer.train()


model.save_pretrained(OUT_DIR)
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

#Zip the trained model for download
zip_filename = "trained_t5_model.zip"
shutil.make_archive(zip_filename.replace(".zip", ""), 'zip', OUT_DIR)

#Inference

from transformers import T5ForConditionalGeneration , T5Tokenizer
import glob

# Unzip the uploaded model
!unzip trained_t5_model.zip -d /content/

# Load the model and tokenizer
model_path = "/content/results_t5base"  # Change if unzipped elsewhere
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

def summarize_text(text, model, tokenizer, max_length=512, num_beams=5):
    # Preprocess the text
    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True
    )

    # Generate the summary
    summary_ids = model.generate(
        inputs,
        max_length=100,
        num_beams=num_beams,
        # early_stopping=True,
    )

    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

test_text = """Budget to set scene for election..Gordon Brown will seek to put the economy at the centre of Labour's bid for a third term in power when he delivers his ninth Budget at 1230 GMT. He is expected to stress the importance of continued economic stability, with low unemployment and interest rates. The chancellor is expected to freeze petrol duty and raise the stamp duty threshold from £60,000. But the Conservatives and Lib Dems insist voters face higher taxes and more means-testing under Labour...Treasury officials have said there will not be a pre-election giveaway, but Mr Brown is thought to have about £2bn to spare...- Increase in the stamp duty threshold from £60,000. - A freeze on petrol duty. - An extension of tax credit scheme for poorer families. - Possible help for pensioners The stamp duty threshold rise is intended to help first time buyers - a likely theme of all three of the main parties' general election manifestos. Ten years ago, buyers had a much greater chance of avoiding stamp duty, with close to half a million properties, in England and Wales alone, selling for less than £60,000. Since then, average UK property prices have more than doubled while the starting threshold for stamp duty has not increased. Tax credits As a result, the number of properties incurring stamp duty has rocketed as has the government's tax take. The Liberal Democrats unveiled their own proposals to raise the stamp duty threshold to £150,000 in February...The Tories are also thought likely to propose increased thresholds, with shadow chancellor Oliver Letwin branding stamp duty a "classic Labour stealth tax". The Tories say whatever the chancellor gives away will be clawed back in higher taxes if Labour is returned to power. Shadow Treasury chief secretary George Osborne said: "Everyone who looks at the British economy at the moment says there has been a sharp deterioration in the public finances, that there is a black hole," he said. "If Labour is elected there will be a very substantial tax increase in the Budget after the election, of the order of around £10bn."..But Mr Brown's former advisor Ed Balls, now a parliamentary hopeful, said an examination of Tory plans for the economy showed there would be a £35bn difference in investment by the end of the next parliament between the two main parties. He added: "I don't accept there is any need for any changes to the plans we have set out to meet our spending commitments."..For the Lib Dems David Laws said: "The chancellor will no doubt tell us today how wonderfully the economy is doing," he said. "But a lot of that is built on an increase in personal and consumer debt over the last few years - that makes the economy quite vulnerable potentially if interest rates ever do have to go up in a significant way." SNP leader Alex Salmond said his party would introduce a £2,000 grant for first time buyers, reduce corporation tax and introduce a citizens pension free from means testing. Plaid Cymru's economics spokesman Adam Price said he wanted help to get people on the housing ladder and an increase in the minimum wage to £5.60 an hour."""

summary = summarize_text(test_text, model, tokenizer)
print("Generated Summary:", summary)


from google.colab import files
files.download(zip_filename)