import json
from utils import NltkUtils
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ChatBotDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CreateDataset():
    def __init__(self,
                 batchSize=32,
                 shuffle=False) -> None:
        self.utils = NltkUtils()

        self.load_intents()

        self.batchSize = batchSize
        self.shuffle = shuffle
        self.allWords = []
        self.tags = []
        self.xy = []
        self.ignoreWords = ['?', '.', ',', '!']
    
    def load_intents(self):
        with open('intents.json', 'r') as f:
            self.intents = json.load(f)
    
    def preprocessData(self):
        for intent in self.intents['intents']:
            tag = intent['tag']
            self.tags.append(tag)
            for pattern in intent['patterns']:
                w = self.utils.tokenize(pattern)
                self.allWords.extend(w)
                self.xy.append((w, tag))

        self.allWords = [self.utils.stem(w) for w in self.allWords if w not in self.ignoreWords]
        self.allWords = sorted(set(self.allWords))
        self.tags = sorted(set(self.tags))
    
    def getTrainingData(self):
        X_train = []
        y_train = []
        self.preprocessData()
        for (pattern_sentence, tag) in self.xy:
            bag = self.utils.bagOfWords(pattern_sentence, self.allWords)
            X_train.append(bag)
            label = self.tags.index(tag)
            y_train.append(label)
        return np.array(X_train), np.array(y_train)  
    
    def getDataLoaders(self):
        X_train, y_train = self.getTrainingData()
        train_loader = DataLoader(ChatBotDataset(X_train, y_train), 
                                  self.batchSize, 
                                  shuffle=True)

        return train_loader, self.allWords, self.tags
