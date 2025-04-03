import torch
import tiktoken
import config as c
from model import GPT2
from torch.optim import Adam
import torch.nn.functional as F

with open('./input.txt') as file:
    input = file.read()

# Creating all the encodings as once
enc = tiktoken.get_encoding('gpt2')
tokens = torch.tensor(enc.encode(input)).type('torch.LongTensor')
decoded = enc.decode(list(tokens))
assert input == decoded

# Implementing the Dataset logic on a smaller scale
x = tokens[:32].reshape(c.batch_size, -1).to(c.device)
y = tokens[1:33].reshape(c.batch_size, -1).to(c.device)

# Initializing the model
model = GPT2()
model.to(c.device)
optim = Adam(model.parameters(), lr = 3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Creating the training loop
for i in range(12):
    logits = model(x)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)

out = model(x)

