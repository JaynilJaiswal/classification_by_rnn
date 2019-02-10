from data import *
from rnn import *

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

import random

criterion = nn.NLLLoss()
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

learning_rate = 0.005 

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()



import time
import math
from random import shuffle
n_iters = 10
print_every = 2000
plot_every = 2000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
shuffle(train_data)
for epoch in range(1, n_iters + 1):
    iter=0
    for example in train_data:
        category_tensor=torch.tensor([all_categories.index(example[0])],dtype=torch.long)
        line_tensor=lineToTensor(example[1])
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss
        iter+=1

        if iter % print_every == 0:
            print('epoch %d %d%% (%s) %.4f' % (epoch, ( iter / len(train_data) )* 100, timeSince(start), loss))

    # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0


torch.save(rnn,'classification.pt')