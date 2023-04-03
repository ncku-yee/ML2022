import json
import numpy as np
import random
import torch
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset 
from transformers import BertForQuestionAnswering, BertTokenizerFast
import transformers
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

parser = ArgumentParser()
parser.add_argument("-p",
                    "--path",
                    type=str,
                    default="macbert",
                    help="Model save directory, path will be like ./models/{PATH}")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

""" Fix random seed for reproducibility """
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)

""" Read json file """
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

""" QA Dataset """
class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs, max_seq_len=512, doc_stride=256):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = max_seq_len
        self.max_question_len = 40
        self.max_paragraph_len = max_seq_len - (1 + self.max_question_len + 1 + 1)
        
        ##### TODO: Change value of doc_stride #####
        self.doc_stride = doc_stride       # Overlapping

        

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn

        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # A single window is obtained by slicing the portion of paragraph containing the answer
            mid = int(random.uniform(answer_start_token, answer_end_token))
            # mid = int((answer_start_token + answer_end_token) // (2 + random.uniform(-1, 1)))
            # mid = (answer_start_token + answer_end_token) // 2
            paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            
            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]
            
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # Pad sequence and obtain inputs to model 
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask

def evaluate(data, output, doc_stride, paragraph, paragraph_offset, topk=5):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''                         # Prediction answer.
    max_prob = float('-inf')            # Maximum probability of logits.
    num_of_windows = data[0].shape[1]   # Total windows.
    window = 0                          # Index of window with max probability.
    win_context = None                  # Ids list of window.
    start_i, end_i = 0, 0               # (Start, End) of window with max probability.
    new_start, new_end = 0, 0           # New (start, end) pair in original paragraph.

    for k in range(num_of_windows):
        # Base index after the Question [CLS] Question? [SEP]
        paragraph_base = list(data[0][0][k]).index(102) + 1
        # Last index of [SEP] in [SEP] paragraph... [SEP]
        paragraph_sep = len(list(data[0][0][k])) - 1 - list(data[0][0][k])[::-1].index(102)

        # Keep the most probable top k start position / end position
        start_probs, start_indices = torch.topk(output.start_logits[k], topk, dim=0)
        end_probs, end_indices = torch.topk(output.end_logits[k], topk, dim=0)

        # Probability of answer is calculated as sum of start_prob and end_prob
        # (start_index, end_index) must in the range of [paragraph_base, paragraph_sep)
        # Length of answer is less than 30
        prob = float('-inf')
        for start_prob, start_idx in zip(start_probs, start_indices):
            for end_prob, end_idx in zip(end_probs, end_indices):
                if (start_idx <= end_idx) and (end_idx - start_idx <= 30) and (start_idx >= paragraph_base) and (end_idx < paragraph_sep):
                    if start_prob + end_prob > prob:
                        prob = start_prob + end_prob
                        start_index, end_index = start_idx, end_idx

        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
            win_context = data[0][0][k]
            start_i, end_i = start_index - paragraph_base, end_index + 1 - paragraph_base
            window = k

    # '[UNK]' in prediction answer.
    # if '[UNK]' in answer:
    #     print(f"[UNK] in prediction answer\n{answer}")
    #     print(f"Prediction range: [{start_i}, {end_i}) in window {window}-th paragraph")

    # Calculate the (start, end) pair corresponding to original paragraph.
    for idx in range(start_i + window * doc_stride, end_i + window * doc_stride):
        sub_start, sub_end = paragraph_offset[idx]
        if idx == start_i + window * doc_stride: 
            new_start = sub_start
        if idx == end_i + window * doc_stride - 1:
            new_end = sub_end

    if '「' in paragraph[new_start:new_end] and paragraph[new_end - 1] != '」':
        new_end += 1
    elif paragraph[new_start] != '「' and '」' in paragraph[new_start:new_end]:
        new_start -= 1
    elif '《' in paragraph[new_start:new_end] and paragraph[new_end - 1] != '》':
        new_end += 1
    elif paragraph[new_start] != '《' and '》' in paragraph[new_start:new_end]:
        new_start -= 1

    # if '[UNK]' in answer:
    #     print(f"Original answer\n{paragraph[new_start:new_end]}")

    # Get answer from original paragraph.
    answer = paragraph[new_start:new_end]
    return answer


# Change "fp16_training" to True to support automatic mixed precision training (fp16)	
fp16_training = True

if fp16_training:
    os.system("pip install accelerate==0.2.0")
    from accelerate import Accelerator
    accelerator = Accelerator(fp16=True)
    device = accelerator.device

# Load pretrained model and tokenizer.
model = BertForQuestionAnswering.from_pretrained("hfl/chinese-macbert-large").to(device)
tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-macbert-large")

# Read dataset
train_questions, train_paragraphs = read_data("hw7_train.json")
dev_questions, dev_paragraphs = read_data("hw7_dev.json")
test_questions, test_paragraphs = read_data("hw7_test.json")

# Tokenize Dataset
train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)



""" Configurations """
max_seq_len = 512
doc_stride = 256
topk = 5
train_batch_size = 8
num_epoch = 3
validation = True
logging_step = 1000
learning_rate = 5e-5
accum_steps = 8
model_save_dir = f"./models/{args.path}"
debug = False


# DataLoader
train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized, max_seq_len=max_seq_len, doc_stride=doc_stride)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized, max_seq_len=max_seq_len, doc_stride=doc_stride)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized, max_seq_len=max_seq_len, doc_stride=doc_stride)

# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

# Debugging
if debug:
    print("Evaluating Dev Set ...")
    model.eval()
    with torch.no_grad():
        dev_acc = 0
        for i, data in enumerate(tqdm(dev_loader)):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                    attention_mask=data[2].squeeze(dim=0).to(device))
            # prediction is correct only if answer text exactly matches
            predict_ans = evaluate(data, output, doc_stride, dev_paragraphs[dev_questions[i]['paragraph_id']], dev_paragraphs_tokenized[dev_questions[i]['paragraph_id']].offsets, topk)
            golden_ans = dev_questions[i]["answer_text"]
            if predict_ans != golden_ans:
                print(f"Question: {dev_questions[i]['question_text']}\nGolden Answer: {golden_ans}")
                print(f"Your   Answer: {predict_ans}")
            dev_acc += predict_ans == golden_ans
        print(f"Validation | acc = {dev_acc / len(dev_loader):.3f}")

# Training
total_steps = len(train_loader) * num_epoch // accum_steps   # Total number of training steps

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

if fp16_training:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 

model.train()

print("Start Training ...")
best_acc = 0
lr = [optimizer.param_groups[0]["lr"]]
for epoch in range(num_epoch):
    step = 1
    train_loss, train_acc = 0, 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        # Load all data into GPU
        data = [i.to(device) for i in data]
        
        # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
        # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
        output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

        # Choose the most probable start position / end position
        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)
        
        # Prediction is correct only if both start_index and end_index are correct
        train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
        train_loss += output.loss

        # Normalize loss to account for batch accumulation
        loss =  output.loss / accum_steps 

        if fp16_training:
            accelerator.backward(loss)
        else:
            loss.backward()
        
        # Gradient accumulation(Update weights)
        if ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            lr.append(optimizer.param_groups[0]["lr"])

        step += 1

        ##### TODO: Apply linear learning rate decay #####
        
        
        # Print training loss and accuracy over past logging step
        if step % logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
            train_loss = train_acc = 0

    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
                # prediction is correct only if answer text exactly matches
                dev_acc += evaluate(data, output, doc_stride, dev_paragraphs[dev_questions[i]['paragraph_id']], dev_paragraphs_tokenized[dev_questions[i]['paragraph_id']].offsets, topk) == dev_questions[i]["answer_text"]
            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")

        if dev_acc / len(dev_loader) > best_acc:
            # Save a model and its configuration file to the directory 「saved_model」 
            # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
            # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
            print(f"Saving Model with Epoch {epoch + 1}...")
            model.save_pretrained(model_save_dir)
            best_acc = dev_acc / len(dev_loader)

        model.train()

# Plot the learning rates.
plt.figure(figsize=(15, 5))
plt.plot(np.arange(1, len(lr)+1), lr)