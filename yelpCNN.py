from functionClass import *
from gensim.models import Word2Vec
import torch
device = 'cpu'

rateReviewTrainList, rateReviewTestList, maxListCount = dataRead()
y_targetList = y_targetFun(rateReviewTrainList)

w2vMyModel = Word2Vec.load('./w2vTrainedModel/w2vTrained.model')


textCNNmodel = TextCnn(w2vMyModel)
textCNNmodel.to(device=device)
lossFunction = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(textCNNmodel.parameters(), lr = 0.001)

reviewLiList = word2VecFun(rateReviewTrainList, maxListCount)
textCNNmodel.train()
numEpochs = 1

for epoch in range(numEpochs):
    print("Epoch-->", epoch)
    trainLoss = 0
    for index, textList in enumerate(reviewLiList):
        textCNNmodel.zero_grad()
        print(textList)
        trainIndexList = tensorFromLine(textList, maxListCount, w2vMyModel)
        y_expected = textCNNmodel(trainIndexList)

        loss = lossFunction(y_expected, y_targetList[index])
        trainLoss += loss.item()
        loss.backward()

        optimizer.step()

testReviewLiList = word2VecFun(rateReviewTestList, maxListCount)
test_x_index = testReviewLiList[0]
test_x_index = tensorFromLine(test_x_index, maxListCount, w2vMyModel)
textCNNmodel.eval()
y_predic = textCNNmodel(test_x_index)
print(y_predic)


