import json
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from random import shuffle
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def dataRead():
    rateReviewList = []
    with open('review1000th.json') as file:
        porter_stemmer = PorterStemmer()
        lineCount = 0
        maxList = 0
        while True:
            line = file.readlines(1)
            if not line:
                break
            if lineCount == 10:
                break
            jsonLine = json.loads(line[0])

            if jsonLine["stars"] <= 1:
                sentiNum = 0
            elif jsonLine["stars"] == 3:
                sentiNum = 1
            else:
                sentiNum = 2
            noStopWords = remove_stopwords(jsonLine['text'])
            stemWords = porter_stemmer.stem(noStopWords)
            tokenWords = simple_preprocess(stemWords, deacc=True)
            if len(tokenWords) > maxList:
                maxList = len(tokenWords)
            newDic = {}
            newDic['rate'] = sentiNum
            newDic['reviewTxt'] = tokenWords
            rateReviewList.append(newDic)

            lineCount = lineCount + 1
    shuffle(rateReviewList)

    splitPosi = int(len(rateReviewList)*0.8)
    rateReviewTrainList = rateReviewList[0:splitPosi]
    rareReviewTestList = rateReviewList[splitPosi:]
    return rateReviewTrainList, rareReviewTestList, maxList

def word2VecFun(dataList, maxListCt):
    for dic in dataList:
        stemList = dic['reviewTxt']
        if len(stemList) < maxListCt:
            for ele in range(maxListCt - len(stemList)):
                stemList.append('pad')

    reviewDwList = [ele['reviewTxt'] for ele in dataList]

    # print("training w2v.....")
    # w2v_model = Word2Vec(reviewDwList, size=10, workers=3, window=3, sg=1)
    # w2v_model.save('./w2vTrainedModel/w2vTrained.model')
    # print("end of training word2Vec---->>>")

    return reviewDwList

def y_targetFun(dicList):
    # print(dicList)
    device = "cpu"
    yList = [torch.tensor([ele['rate']], dtype=torch.long, device=device) for ele in dicList]
    # print(yList)

    return yList


def tensorFromLine(lineList, maxListCount, w2vMyModel):
    w2vMyModel = Word2Vec.load('./w2vTrainedModel/w2vTrained.model')
    padIndex = w2vMyModel
    trainIndexList = [padIndex for i in range(maxListCount)]

    for i, word in enumerate(lineList):
        if word not in w2vMyModel.wv.vocab:
            trainIndexList[i] = 0
        else:
            trainIndexList[i] = w2vMyModel.wv.vocab[word].index
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    return torch.tensor(trainIndexList, dtype=torch.long, device=device).view(1, -1)

class TextCnn(nn.Module):
    def __init__(self, myW2Vmodel):
        super(TextCnn, self).__init__()
        weights = myW2Vmodel.wv
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors),
                                                   padding_idx=myW2Vmodel.wv.vocab['pad'].index)
        NUM_FILTERS = 9
        EMBEDDING_SIZE = 10
        window_sizes=(1,2,3,5)

        self.convs = nn.ModuleList([
                                   nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0))
                                   for window_size in window_sizes
        ])

        num_classes = 3
        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)

    def forward(self, lineList):
        # print(lineList)
        x = self.embedding(lineList)  # [B, T, E]

        # Apply a convolution + max_pool layer for each window size
        x = torch.unsqueeze(x, 1)
        xs = []
        for conv in self.convs:
            x2 = torch.tanh(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)

        # FC
        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        probs = F.softmax(logits, dim=1)

        return probs