import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
import re
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Import the csv into pandas dataframe and add the headers
df = pd.read_csv('Vendors dataset.csv')
print(df.head())


class TextCleaner():
    def __init__(self):
        pass

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
import phonenumbers
def find_PhoneNumber(text):
    for match in phonenumbers.PhoneNumberMatcher(text, "US"):
        result=phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
        return result

import pyap
def find_address(text):
    address=pyap.parse(text,country="US")
    return address

cleaner = TextCleaner()
df['cleaned_text']=df['Bio'].apply(lambda x: cleaner.clean_text (str(x)) if pd.notnull(x) else x)
df['Phone Number'] = df['Bio'].apply(lambda x: find_PhoneNumber(x) if pd.notna(x) else None)
df['Address'] = df['Bio'].apply(lambda x: find_address(x) if pd.notna(x) else None)


# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')



class Triage(Dataset):
    def __init__(self, dataframe,labels, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels=labels

    def __getitem__(self, index):
        title=self.data[index]
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.labels[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len

# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_dataset=df['cleaned_text'].sample(frac=train_size,random_state=200)
test_dataset=df['cleaned_text'].drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)
labels=df['cluster']

print("FULL Dataset: {}".format(df['cleaned_text'].shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = Triage(train_dataset,labels, tokenizer, MAX_LEN)
testing_set = Triage(test_dataset,labels, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


model = DistillBERTClass()
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

# Function to calcuate the accuracy of the model

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return

for epoch in range(EPOCHS):
    train(epoch)


def valid(model, testing_loader):
    model.eval()
    n_correct = 0;
    n_wrong = 0;
    total = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu

print('This is the validation section to print the accuracy and see how it performs')
print('Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch')

for data in testing_loader:
    outputs = model(data['ids'],data['mask'])
    print(f"Outputs shape: {outputs.shape}")
    print('Target',data['targets'].shape)


acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)

# Saving the files for re-use

output_model_file = './models/pytorch_distilbert_news.bin'
output_vocab_file = './models/vocab_distilbert_news.bin'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')
print('This tutorial is completed')


#loading the saved model to make predictions
new_df=pd.read_csv('new vendors.csv')

model = DistillBERTClass()
model.load_state_dict(torch.load(output_model_file, map_location=device))
model.to(device)
model.eval()

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(output_vocab_file)

#preprocess the input text
def preprocess_text(text, tokenizer, max_len):
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_token_type_ids=True,
        truncation=True
    )
    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
    return ids, mask

cleaner = TextCleaner()
new_df['cleaned_text']=new_df['Bio'].apply(lambda x: cleaner.clean_text (str(x)) if pd.notnull(x) else x)
new_df['Phone Number'] = new_df['Bio'].apply(lambda x: find_PhoneNumber(x) if pd.notna(x) else None)
new_df['Address'] = new_df['Bio'].apply(lambda x: find_address(x) if pd.notna(x) else None)

#perform inference
processed_data = new_df['cleaned_text']=.apply(lambda x: preprocess_text (x,tokenizer, max_len) if pd.notnull(x) else x)

predictions = []
for text in new_df['cleaned_text']:
    ids, mask = preprocess_text(text, tokenizer, MAX_LEN)
    with torch.no_grad():
        outputs = model(ids, mask)
        predicted_class = torch.argmax(outputs, dim=1).item()
        predictions.append(predicted_class)

# Add predictions to the dataframe
new_df['cluster'] = predictions
print(new_df.head())

result=pd.concat([df,new_df], axis=0,ignore_index=True)
result.to_csv('Vendors dataset.csv', index=False)
print("Predictions saved to 'Vendors dataset.csv'")
