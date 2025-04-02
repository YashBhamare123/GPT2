import torch
from model import GPT2
import torch.nn.functional as F


# Function to load pretrained weights into model
def from_pretrained(model_type):
    """Loads pretrained GPT-2 model weights from huggingface"""
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
        'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    }[model_type]
    config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
    # create a from-scratch initialized minGPT model
    model = GPT2()
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # copy while ensuring all the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model

# Function that creates tokens and sends them into the model for generation
# Returns an output string
def generate_output(max_length, temp, input_seq, enc):
    # Start with the encoded input sequence as a list of token IDs
    token_list = enc.encode(input_seq)

    # Generate until the total length reaches max_length
    while len(token_list) < max_length:
        # Convert token list to tensor with batch dimension (1, seq_len)
        tokens = torch.tensor(token_list).unsqueeze(0).type("torch.IntTensor")

        # Get model output (logits) - assuming shape (1, seq_len, vocab_size)
        out = model2(tokens)

        # Take logits for the last position to predict the next token
        logits = out[0, -1, :]  # Shape: (vocab_size,)

        # Apply temperature and compute probabilities
        logits = logits / temp
        probs = F.softmax(logits, dim=-1)

        # Sample the next token ID
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Append to the token list
        token_list.append(next_token)

    # Decode the full list of tokens to a string
    out_seq = enc.decode(token_list)
    return out_seq

# Function that returns the available device
def get_default_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


if __name__ == '__main__':
    model2 = GPT2()
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    def generate(max_length, temp, input_seq, enc):
        out_seq = input_seq
        tokens = torch.tensor(enc.encode(input_seq)).type("torch.IntTensor").unsqueeze_(dim = 0)
        while tokens.size()[-1] < max_length:
            tokens = torch.tensor(enc.encode(input_seq)).type("torch.IntTensor").unsqueeze_(dim = 0)
            out = model2(tokens)
            out = F.softmax(out, dim = -1)
            indice = torch.argmax(out, dim = -1)[0][-1]
            word = enc.decode([int(indice)])
            out_seq = out_seq + word
            input_seq = out_seq
        return out_seq


    import torch
    import torch.nn.functional as F


    def generate1(max_length, temp, input_seq, enc):
        # Start with the encoded input sequence as a list of token IDs
        token_list = enc.encode(input_seq)

        # Generate until the total length reaches max_length
        while len(token_list) < max_length:
            # Convert token list to tensor with batch dimension (1, seq_len)
            tokens = torch.tensor(token_list).unsqueeze(0).type("torch.IntTensor")

            # Get model output (logits) - assuming shape (1, seq_len, vocab_size)
            out = model2(tokens)

            # Take logits for the last position to predict the next token
            logits = out[0, -1, :]  # Shape: (vocab_size,)

            # Apply temperature and compute probabilities
            logits = logits / temp
            probs = F.softmax(logits, dim=-1)

            # Sample the next token ID
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Append to the token list
            token_list.append(next_token)

        # Decode the full list of tokens to a string
        out_seq = enc.decode(token_list)
        return out_seq
    print(generate1(205, 0.9, "Once upon a time,", enc))


