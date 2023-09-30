import torch.nn as nn
import torch
from absl import flags, app
import random
import json

from model import NeuralNetwork
from utils import NltkUtils


class ChatBot:
    def __init__(self,
                 botName="Sam") -> None:
        super().__init__()
        self.botName = botName
        self.utils = NltkUtils()
        self.load_intents()

        self.loadData()
        self.model = NeuralNetwork(inputSize=self.inputSize,
                                   h1Size=self.hiddenSize,
                                   h2Size=self.hiddenSize,
                                   outputSize=self.outputSize)
        self.model.load_state_dict(self.modelState)
        self.model.eval()
    
    def load_intents(self):
        with open('intents.json', 'r') as f:
            self.intents = json.load(f)

    def loadData(self):
        data = torch.load("data.pth")
        self.modelState = data['modelState']
        self.inputSize = data['inputSize']
        self.hiddenSize = data['hiddenSize']
        self.outputSize = data['outputSize']
        self.allWords = data['allWords']
        self.tags = data['tags']
    
    def get_response(self, sentence):
        sentence = self.utils.tokenize(sentence)
        X = self.utils.bagOfWords(sentence, self.allWords)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        else:
            return "I do not understand..."

    def chat(self):
        print("Let's chat! (type 'quit' to exit)")
        while True:
            sentence = input("You: ")
            if sentence == "quit":
                break

            print(f"{self.botName}: {self.get_response(sentence)}")


if __name__ == "__main__":
    bot = ChatBot()
    bot.chat()
