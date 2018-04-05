# A character RNN for classifying names!
# The data is contained in the data -> names directory
# There are 18 text files named as [language].txt
# Each file contains names

from io import open
import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math


# Returns list of all text files
def find_files(path):
    return glob.glob(path)

print(find_files('data/names/*.txt'))

# Basically our entire vocabulary of all the characters
all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

# Covert unicoded string to plain ASCII type
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn' 
        and c in all_letters
    )

# Build a category_lines dictionary which represents 
# a list of names per category
category_lines = {}
all_categories = []

# Read a file line by line and convert unicoded lines to ASCII
def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

# Iterate over all filenames and fill in the category_lines dictionary
# Also we keep track of all the categories in all_categories list
for filename in find_files('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
# Printing first 10 names in Italian category
print(category_lines['Italian'][:10])

# Turning the names into Tensors for further usage
# A single letter is represented by a one-hot encoding
# To make a word we make a 2d matrix of one hot encoding of all the letters
# in that word
# We make a tensor of dimensions (line_length, 1, n_letters)
# Extra dimension is due to the fact that PyTorch want batch_size dim
def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

# Creating the network
# Basic RNN implementation
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
    
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# To run a step of this network we need to pass the input 
# i.e tensor of the current letter and previous hidden state
# which is initially a vector of zeros
# We get back probability of each language as output and a next hidden layer
input = Variable(letter_to_tensor('A'))
hidden = Variable(torch.zeros(1, n_hidden))
output, next_hidden = rnn(input, hidden)

# For efficiency though, rather than using letter to index 
# we use line to index
# Further optimization by precomputing the batches
input = Variable(line_to_tensor('Albert'))
hidden = Variable(torch.zeros(1, n_hidden))
output, next_hidden = rnn(input[0], hidden)
print(output)

# Training
def category_from_output(output):
    _, top_i = output.data.topk(1)
    print(top_i)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

print(category_from_output(output))

# A quick way to get training example
def random_choice(l):
    return l[random.randint(0, len(l) - 1)]

def random_training_example():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = random_training_example()
    print(category, line)

# Lost function
criterion = nn.NLLLoss()
learning_rate = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

n_iter = 200000
print_every = 5000
plot_every = 1000

# Keep track of loss for plotting later on
current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iter + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iter * 100, time_since(start), loss, line, guess, correct))

     # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# Save the trained model
torch.save(rnn, 'model')

