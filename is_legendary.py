#!/usr/bin/env python
# coding: utf-8

# In[2]:


# For manipulating and analyzing data
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns

# For Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# For making the model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

# For evaluating the model
from sklearn.metrics import confusion_matrix

# For visualization
from PIL import Image


# In[3]:


dataset = pd.read_csv('pokemon.csv')
print(dataset.shape)
ends = pd.concat([dataset.head(), dataset.tail()])
ends


# In[5]:


# How does the class label distribution look?
# We want a fairly even distribution of 0 to 1 labels (non-legendary to legendary in this case)
sns.countplot(x='is_legendary', data=dataset)


# In[6]:


# There are many less legendaries than non-legendaries, so it's an imbalanced dataset.


# In[7]:


# Build the dataframe with only columns where dtype is not object
dataset_cleaned = dataset.select_dtypes(exclude=['object'])
# Drop specific columns that intuitively are unhelpful
dataset_cleaned = dataset_cleaned.drop(columns=dataset_cleaned.filter(regex='^against').columns)
dataset_cleaned = dataset_cleaned.drop(['pokedex_number', 'generation'], axis=1)
# Drop percentage_male because it's null for most legenedaries
dataset_cleaned = dataset_cleaned.drop(['percentage_male'], axis=1)
# Drop rows with nan values
dataset_cleaned = dataset_cleaned.dropna()


# In[8]:


# Check how many legendaries were dropped
print(dataset.loc[dataset['is_legendary'] != 0].shape)
print(dataset_cleaned.loc[dataset_cleaned['is_legendary'] != 0].shape)


# In[9]:


# Print current state of the dataset
print(dataset_cleaned.shape)
dataset_cleaned.head()


# In[10]:


sns.countplot(x='is_legendary', data=dataset_cleaned)


# In[11]:


# Split the data into inputs and labels
X = dataset_cleaned.iloc[:, :-1] # all rows, all but last column
Y = dataset_cleaned.iloc[:, -1] # all rows, only last column


# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2^31-1, stratify=Y) # stratify keeps the label distribution roughly equal between training and test sets
print(Y_train.describe())
print()
print(Y_test.describe())


# In[13]:


# Dataframes to numpy
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

X_train[0], Y_train[0]


# In[14]:


# Standardize the data to improve performance of the model (centers the data at 0 with std of 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[15]:


# More imports
import torch


# In[16]:


# Build Dataset class
## Train Data (inputs and labels)
class TrainData(Dataset): # inherit from Dataset class
    def __init__(self, X_data, Y_data):
        self.x_data = X_data
        self.y_data = Y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

## Test Data (just inputs)
class TestData(Dataset): # inherit from Dataset class
    def __init__(self, X_data):
        self.x_data = X_data

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return len(self.x_data)

train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
test_data = TestData(torch.FloatTensor(X_test))


# In[17]:


train_data[0], test_data[0]


# The Model

# In[18]:


### Hyperparameters
# for the model
HIDDEN_SIZE = 64
# for training
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[19]:


# Build Dataloader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_data, batch_size=1)


# In[20]:


# Define model
class PokemonModel(nn.Module):
   def __init__(self, input_size, hidden_size):
       super(PokemonModel, self).__init__()
       self.l1 = nn.Linear(input_size, hidden_size)
       self.relu1 = nn.ReLU()
       self.batchnorm1 = nn.BatchNorm1d(hidden_size)
       self.l2 = nn.Linear(hidden_size, hidden_size)
       self.relu2 = nn.ReLU()
       self.batchnorm2 = nn.BatchNorm1d(hidden_size)
       self.dropout = nn.Dropout(0.1)
       self.out = nn.Linear(hidden_size, 1)

       self.layers = [
           self.l1,
           self.relu1,
           self.batchnorm1,
           self.l2,
           self.relu2,
           self.batchnorm2,
           self.dropout,
           self.out
       ]

   def forward(self, x):
       for layer in self.layers:
           x = layer(x)
       return x


# In[21]:


# Initialize model
model = PokemonModel(input_size=X_train.shape[1], hidden_size=HIDDEN_SIZE)
model.to(device)
# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
model


# In[22]:


# Define the accuracy metric
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


# In[23]:


model.train() # Put model into 'train mode', which includes dropout and batchnorm layers. Regularly, they are excluded
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()

        Y_pred = model(X_batch)

        loss = criterion(Y_pred, Y_batch.unsqueeze(1))
        acc = binary_acc(Y_pred, Y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')


# In[24]:


Y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        Y_test_pred = model(X_batch)
        Y_test_pred = torch.sigmoid(Y_test_pred)
        Y_pred_tag = torch.round(Y_test_pred)
        Y_pred_list.append(Y_pred_tag.cpu().numpy())
    
Y_pred_list = [a.squeeze().tolist() for a in Y_pred_list]


# In[25]:


confusion_matrix(Y_test, Y_pred_list)


# Lets look at some specific cases

# In[1]:


# Import the pictures
import kagglehub

# Download latest version
path = kagglehub.dataset_download("arenagrenade/the-complete-pokemon-images-data-set")

print("Path to dataset files:", path)


# In[27]:


# Get a legendary pokemon from the dataframe
# pokemon = dataset.sample()                                # sample any pokemon
pokemon = dataset[dataset['is_legendary'] == 1].sample()    # sample a legendary pokemon
# Get the pokemon name
pokemon_name = pokemon['name'].values[0].lower()

# Clean example as before
pokemon = pokemon[dataset_cleaned.columns].values[0][:-1]
pokemon = scaler.transform(pokemon.reshape(1, -1))
pokemon = torch.FloatTensor(pokemon)
# Predict
model.eval()
with torch.no_grad():
    Y_pred = model(pokemon.to(device))
    Y_pred = torch.sigmoid(Y_pred)

img = Image.open(f'pokemon_images/{pokemon_name}.png')
print(f'Pokemon: {pokemon_name}')
print(f'Legendary: {Y_pred.item():.3f}')
img


# In[ ]:




