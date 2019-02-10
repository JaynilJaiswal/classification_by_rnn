from rnn import *
from data import *

from torch.autograd import Variable
import sys

rnn = torch.load('classification.pt')

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(line, n_predictions=3):
    #output = evaluate(Variable(lineToTensor(line)))
    output= evaluate(Variable(lineToTensor(line)))
    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions


num_correct=0
for example in test_data:
    line_tensor=lineToTensor(example[1])
    category=example[0]
    output=evaluate(line_tensor)
    guess,guess_i=categoryFromOutput(output)
    if guess == category:
        num_correct+=1

accuracy=num_correct*100/len(test_data)
print('accuracy is %.4f%%' % (accuracy))