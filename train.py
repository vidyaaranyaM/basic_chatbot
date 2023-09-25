from create_dataset import CreateDataset
from model import NeuralNetwork
import torch.nn as nn
import torch
from absl import flags, app


FLAGS = flags.FLAGS
flags.DEFINE_integer("batchSize", "8", "batch size")
flags.DEFINE_float("learningRate", "0.001", "learning rate")
flags.DEFINE_integer("numEpochs", "1000", "number of epochs")
flags.DEFINE_integer("hiddenSize", "8", "hidden size of nn model")
flags.DEFINE_bool("shuffle", "True", "shuffle the dataset or not")
flags.DEFINE_string("filePath", "data.pth", "file where data is stored")


class Train:
    def __init__(self,
                 batchSize=32,
                 learningRate=0.001,
                 numEpochs=1000,
                 hiddenSize=8,
                 shuffle=False) -> None:
        
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.shuffle = shuffle
        self.hiddenSize = hiddenSize
        
        self.dataset = CreateDataset(batchSize=self.batchSize,
                                     shuffle=self.shuffle)
        self.trainLoader, self.allWords, self.tags = self.dataset.getDataLoaders()

        self.inputSize, self.outputSize = self.getDataSize()
        self.model = NeuralNetwork(inputSize=self.inputSize,
                                   h1Size=self.hiddenSize,
                                   h2Size=self.hiddenSize,
                                   outputSize=self.outputSize)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.learningRate)
    
    def getDataSize(self):
        words, _ = next(iter(self.trainLoader))
        return words.shape[1], 7

    def trainOneEpoch(self):
        avg_loss = 0
        count = 0
        for batch in self.trainLoader:
            words, labels = batch
            outputs = self.model.forward(words)

            loss = self.loss(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()
            count += labels.shape[0]
        return avg_loss / count
    
    def saveData(self):
        data = {"modelState": self.model.state_dict(),
                "inputSize": self.inputSize,
                "hiddenSize": self.hiddenSize,
                "outputSize": self.outputSize,
                "allWords": self.allWords,
                "tags": self.tags}
        torch.save(data, FLAGS.filePath)
        print("Saved Data")

    def train(self):
        for epoch in range(self.numEpochs):
            trainLoss = self.trainOneEpoch()
            if epoch % 100 == 0:
                print("epoch: ", epoch, "loss: ", trainLoss)
        self.saveData()


def main(_):
    train = Train(batchSize=FLAGS.batchSize, 
                  learningRate=FLAGS.learningRate,
                  numEpochs=FLAGS.numEpochs,
                  hiddenSize=FLAGS.hiddenSize,
                  shuffle=FLAGS.shuffle
                  )
    train.train()

if __name__ == "__main__":
    app.run(main)
