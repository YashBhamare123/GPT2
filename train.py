import torch
import tiktoken
import config as c
from model import GPT2
from torch.optim import AdamW
import torch.nn.functional as F

with open('./input.txt') as file:
    input = file.read()

# Creating all the encodings as once
enc = tiktoken.get_encoding('gpt2')
tokens = torch.tensor(enc.encode(input)).type('torch.LongTensor')

# Implementing the Dataset logic on a smaller scale
x = tokens[:1024].view(c.batch_size, -1).to(c.device)
y = tokens[1:1025].view(c.batch_size, -1).to(c.device).to(torch.long)
# Initializing the model
model = GPT2()
model.to(c.device)
optim = AdamW(model.parameters(), lr = 3e-4)

# Creating the training loop
for i in range(12):
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)

out = model(x)

