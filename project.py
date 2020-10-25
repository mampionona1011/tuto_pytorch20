import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator, Iterator
import torch.nn as nn
import torch.optim as optim

#tokenizer function
tokenizer = lambda x: x.split()

a = 'Hello World!'
print(a)
print('Tokenization: ',tokenizer(a))

#define fields
TEXT = Field(sequential = True, tokenize = tokenizer, lower = True)
LABEL = Field(sequential = False, use_vocab = False, dtype=torch.float)

fields = [("question_text", TEXT),("label", LABEL)]

#load datasets
train_data = TabularDataset(path = 'data/train.csv',format = 'csv',fields = fields,skip_header = True)
valid_data = TabularDataset(path = 'data/validation.csv',format = 'csv',fields = fields,skip_header = True)
test_data = TabularDataset(path='data/test.csv',format='csv', skip_header=True, fields=fields)


##print(len(train_data))
##print(len(valid_data))
##print(len(test_data))

#build vocabulary 
TEXT.build_vocab(train_data)

print('Vocabualry size: ', len(TEXT.vocab))
print('First example: ', vars(train_data.examples[0]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#set batch size
BATCH_SIZE = 64

#load iterator
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE ,
    sort_key = lambda x: len(x.question_text),
    sort_within_batch = True,
    device = device)

class Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()

        #defining the layers

        #embedding layer
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        #rnn layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        
        #linear layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_batch):

        embedded = self.embedding(input_batch)
                
        _, hidden = self.rnn(embedded)
                
        return self.fc(hidden.squeeze(0))
    
#define hyper-parameters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 1
LEARNING_RATE = 0.001

#instantiate the model
model = Model(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
print(model)

#define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

def get_accuracy(pred, y):
    
    pred = torch.round(torch.sigmoid(pred))
    
    correct = (pred == y).float()
    
    accuracy = sum(correct) / len(correct)
    
    return accuracy

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        #forward propagation
        predictions = model(batch.question_text).squeeze(1)
        
        #loss computation
        loss = criterion(predictions, batch.label)
        acc = get_accuracy(predictions, batch.label)

        #backward propagation
        loss.backward()

        #weight optimization
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
         
        
    return epoch_loss / len(iterator),epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():

        for batch in iterator:

            predictions = model(batch.question_text).squeeze(1)

            loss = criterion(predictions, batch.label)
            acc = get_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_model():
    N_EPOCHS = 10

    min_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')
    
        print(f'Epoch: {epoch+1}')
        print(f'\tLoss: {train_loss:.3f} (train)\tAccuracy: {train_acc*100:.2f}% (train)')
        print(f'\tLoss: {valid_loss:.3f} (valid)\tAccuracy: {valid_acc*100:.2f}% (valid)')
        
    #check the results of test dataset    
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Loss: {test_loss:.3f} (test)\tAccuracy: {test_acc*100:.2f}% (test)')
    to_do()


def predict(sentence):
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    sentence = TEXT.preprocess(sentence)
    x = torch.tensor([TEXT.vocab.stoi[w] for w in sentence]).unsqueeze(1)

    pred = model(x)
    pred = torch.round(torch.sigmoid(pred)).item()
    if (pred == 0.0):
        label = 'Positive'
    else:
        label = 'Negative'
    print(label)
    
    
def to_do():
    choice = input("Type (1 or 2):\n1. Training\n2. Predict\n")
    if (choice == '1'):
        train_model()
    else :
        while True:
            text = input("Type your text: (Type Q to quit)\n")
            if (text !='Q'):
                predict(text)
            if (text == 'Q'):
                break;
to_do()
