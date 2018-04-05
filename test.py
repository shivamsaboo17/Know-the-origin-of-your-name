from torch.autograd import Variable
import torch
import string
import torch.nn as nn
import glob

def find_files(path):
    return glob.glob(path)

def load_model(model_name):
    model = torch.load(model_name)
    return model

def get_input():
    name = input()
    return name

def evaluate(name, model):
    line_tensor = Variable(line_to_tensor(name))
    hidden = model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    return output

def get_results(output):
    return category_from_output(output)

def category_from_output(output):
    _, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return category_i

def line_to_tensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for li, letter in enumerate(name):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def letter_to_index(letter):
    return all_letters.find(letter)

def generate_category_list():
        for filename in find_files('data/names/*.txt'):
            category = filename.split('/')[-1].split('.')[0]
            all_categories.append(category)

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

    

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)
all_categories = []
generate_category_list()
print("Loading model...")
model = load_model('model')
print("Model loaded. Enter a name to know about it's origin!")
print("Type exit to stop")
while True:
    name = get_input()
    if name == 'exit':
        break
    output = evaluate(name, model)
    result = get_results(output)
    print(name, "has origin from", all_categories[result])
