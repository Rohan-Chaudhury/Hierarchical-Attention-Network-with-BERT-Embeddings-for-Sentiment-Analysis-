import sys
import getopt
import os
import math
import operator
from tkinter import Y
import numpy as np
import json

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from tqdm import trange


debug=0
n_cpus = os.cpu_count()
use_cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
if debug:
    print("Number of cpus: ", n_cpus)
    print("device: ", device)

use_sentence_level_attention = True
use_word_level_attention = True

class CommonAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CommonAttention, self).__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.contx = nn.Linear(hidden_dim, 1, bias=False).to(device)

    def forward(self, inp):
        
        u = torch.tanh_(self.attn(inp)).to(device)

        a = F.softmax(self.contx(u), dim=1).to(device)
  
        s = (a * inp).sum(1).to(device)
        return a.permute(0, 2, 1), s


class WordLevelAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(WordLevelAttention, self).__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.contx = nn.Linear(hidden_dim, 1, bias=False).to(device)

    def forward(self, inp):
        if use_word_level_attention:
          u = torch.tanh_(self.attn(inp)).to(device)

          a = F.softmax(self.contx(u), dim=1).to(device)
    
          s = (a * inp).sum(1).to(device)
        else:
          s = inp.sum(1).to(device)
          a= torch.zeros(1,1,1).to(device)

        return a.permute(0, 2, 1), s


class SentenceLevelAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SentenceLevelAttention, self).__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.contx = nn.Linear(hidden_dim, 1, bias=False).to(device)

    def forward(self, inp):
        if use_sentence_level_attention:
          u = torch.tanh_(self.attn(inp)).to(device)

          a = F.softmax(self.contx(u), dim=1).to(device)
    
          s = (a * inp).sum(1).to(device)
        else:
          s = inp.sum(1).to(device)
          a= torch.zeros(1,1,1).to(device)

        return a.permute(0, 2, 1), s

class WordAttnNet(nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        embed_dim=50
    ):
        super(WordAttnNet, self).__init__()



        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True).to(device)

        if use_word_level_attention and use_sentence_level_attention:
          self.word_attn = CommonAttention(hidden_dim * 2).to(device)
        else:
          self.word_attn = WordLevelAttention(hidden_dim * 2).to(device)

    def forward(self, X, h_n):
        embed = X
        h_t, h_n = self.rnn(embed, h_n)
        a, s = self.word_attn(h_t)
        return a, s.unsqueeze(1), h_n


class SentAttnNet(nn.Module):
    def __init__(
        self, word_hidden_dim=32, sent_hidden_dim=32, padding_idx=1
    ):
        super(SentAttnNet, self).__init__()

        self.rnn = nn.GRU(
            word_hidden_dim * 2, sent_hidden_dim, bidirectional=True, batch_first=True
        ).to(device)

        if use_word_level_attention and use_sentence_level_attention:
          self.sent_attn  = CommonAttention(sent_hidden_dim * 2).to(device)
        else:
          self.sent_attn = SentenceLevelAttention(sent_hidden_dim * 2).to(device)

    def forward(self, X):
        h_t, h_n = self.rnn(X)
        a, v = self.sent_attn(h_t)
        return a.permute(0,2,1), v


class HierAttNet(nn.Module):
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []
      
  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    super(HierAttNet, self).__init__()
    """network initialization"""
    self.numFolds = 10
    self.batch=20
    self.word_hidden_dim=32
    self.sent_hidden_dim=32
    self.embed_dim=300
    self.padding_idx=1


    self.wordattnnet = WordAttnNet(
        hidden_dim=self.word_hidden_dim,
        embed_dim=self.embed_dim
    )

    self.sentattnnet = SentAttnNet(
        word_hidden_dim=self.word_hidden_dim,
        sent_hidden_dim=self.sent_hidden_dim,
        padding_idx=self.padding_idx,
    )

    self.fc = nn.Linear(self.sent_hidden_dim * 2, 1).to(device)
    # torch.sigmoid(
  def forward(self, X):
      x = X.permute(1, 0, 2, 3)
      word_h_n = nn.init.zeros_(torch.Tensor(2, X.shape[0], self.word_hidden_dim)).to(device)
      # if use_cuda:
      #     word_h_n = word_h_n.cuda()
      # alpha and s Tensor Lists
      word_a_list, word_s_list = [], []
      for sent in x:
          word_a, word_s, word_h_n = self.wordattnnet(sent, word_h_n)
          word_a_list.append(word_a)
          word_s_list.append(word_s)
      # Importance attention weights per word in sentence
      self.sent_a = torch.cat(word_a_list, 1).to(device)

      sent_s = torch.cat(word_s_list, 1).to(device)

      self.doc_a, doc_s = self.sentattnnet(sent_s)
      to_return = self.fc(doc_s)
#       print(to_return.shape)

      return torch.sigmoid(to_return)


  def dataConvert(self, words):
    """
      convert one document into numpy array of word embedding [1, #sent, #word, #glove_embedding(300)]
    """
    max_word=50
    max_sent=20
    sent=torch.zeros(1, max_sent, max_word, 300, device=device, dtype=torch.float32)
    sid=0
    wid=0
    slen=[]

    for word in words:        
      if word =='<eos>':
        sid+=1
        slen.append(wid)
        wid=0
      elif wid<max_word and word.lower() in self.vocab_embedding:
        sent[0,sid,wid,:]=torch.from_numpy(self.vocab_embedding[word.lower()]).to(device)
        wid+=1
      if sid>=max_sent:
        return sent,slen    
    
    if words[-1]!='<eos>':
      slen.append(wid)
           
    return sent,slen

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the HierAttNet classifier 

  def classify(self, words):
    """ TODO
      implement the prediction function of HierAttNet  
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    label={'pos':0,'neg':1}
    x_data,x_lens=self.dataConvert(words)
    x_data=x_data.to(device)
    Y=self(x_data)
#     print("classify")
#     for param in self.parameters():
#       print(param.data)
    if debug:
        print("classifier: ", Y)
    return 'pos' if Y<0.5 else 'neg'

    # Write code here


  
  def train(self, split, iterations):
    """
     * TODO
     * Train your model with dataset in the format of numpy array [x_data,y_data]
     * x_data: int numpy array, dimension=[#document, #sent, #word, #glove_embedding(300)].
     * y_data: int numpy array, dimension: [#document].
     * x_lens: list, stores the list of sentence length in each document.
     * before training, you need to define HierAttNet sub-modules
     * in the HierAttNet class with a deep learning framework.
     * Returns nothing
    """
    label={'pos':0,'neg':1}
    eid=0
    y_data=[]
    x_lens=[]
    for example in split.train:      
      words = example.words  
      x,x_len=self.dataConvert(words)
      x_lens.append(x_len)
      y_data.append(label[example.klass])
      if eid==0:
        x_data=x.detach().clone().to(device)
        eid+=1
      else:          
        x_data=torch.cat((x_data,x), axis=0).to(device)
    y_data=torch.Tensor(y_data).to(device)

    train_set = TensorDataset(
      x_data,
        y_data)
    if debug:
        print(self.batch)
    train_loader = DataLoader(dataset=train_set, batch_size=self.batch, num_workers=0,shuffle=True)


    # Write code here
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
    if debug:
        print("initial_training") 
#     for param in self.parameters():
#       print(param.data)
    # optimizer = torch.optim.AdamW(self.parameters()) 
    for epoch in range(iterations):
      train_steps = len(train_loader)
      running_loss = 0
      with trange(train_steps) as t:
        for batch_idx, (data, target) in zip(t, train_loader):
          t.set_description("epoch %i" % (epoch + 1))
          if use_cuda:
            data, target = data.cuda(), target.cuda()


          optimizer.zero_grad()
          output = self(data)
          loss = loss_fn(output,target.reshape(-1,1))
          if debug:
              print("output: ", output)
              print("target: ", target)
          #backprop
         
          loss.backward()
          optimizer.step()
          # print(self.parameters().data)
          running_loss += loss.item()
          t.set_postfix(avg_loss=running_loss / (batch_idx + 1))


  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords(' <eos> '.join(contents))
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  
  def loadEmbedding(self, embed_path):
    with open(embed_path, 'r') as file:
      self.vocab_embedding=json.load(file)
    for key in self.vocab_embedding.keys():
        self.vocab_embedding[key]=np.array(self.vocab_embedding[key])
    return
   
def test10Fold(args):
  pt = HierAttNet()
  pt.batch=int(args[2])
  iterations = int(args[1])
  pt.loadEmbedding(args[3])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = HierAttNet()
    classifier.vocab_embedding=pt.vocab_embedding
    classifier.batch=pt.batch
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if debug:
          print("original:", example.klass)
      if example.klass == guess:
        
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)
      
      
def main(): 
  (options, args) = getopt.getopt(sys.argv[1:], '')
  test10Fold(args)

if __name__ == "__main__":
    main()
