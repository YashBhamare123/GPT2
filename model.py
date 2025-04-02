import torch
import torch.nn as nn
import config as c
import torch.nn.functional as F

class GPT2(nn.Module):
    def __init__(self):
        super().__init__()

        # Implementing the transformer
        self.transformer = nn.ModuleDict({
            "wte" : nn.Embedding(c.vocab_size, c.embed_dim),
            "wpe" : nn.Embedding(c.seq_length, c.embed_dim),
            "h" : nn.ModuleList([AttentionBlock() for _ in range(c.n_layers)]),
            "ln_f" : nn.LayerNorm(c.embed_dim)
        })

        # Implementing the dot product layer to compute the logits
        self.lm_head = nn.Linear(c.embed_dim, c.vocab_size, bias = False)

        # Implementing parameter sharing
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, token_ids):
        # Creating the token vectors
        B = token_ids.size()[0]
        tokens = self.transformer.wte(token_ids)

        # Creating the positioning vectors
        pos = torch.arange(0, c.seq_length, step = 1).to(c.device)
        token_pos = self.transformer.wpe(pos).repeat([B,1, 1])[:, :token_ids.size()[1], :]
        out = tokens + token_pos

        # Passing the input through attention blocks
        for i in range(c.n_layers):
            out = self.transformer.h[i](out)

        # Creating the probability vectors by taking dot product
        out = self.lm_head(self.transformer.ln_f(out))
        return out


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.embed_dim)
        self.attn = MaskedAttention()
        self.ln_2 = nn.LayerNorm(c.embed_dim)
        self.mlp = MLP()


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MaskedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the query, key, value matrices which can be computed in one matrix multiplication
        self.c_attn = nn.Linear(c.embed_dim, 3*c.embed_dim)
        # Defining the linear transformation that we apply on the concatenated attention heads
        self.c_proj = nn.Linear(c.embed_dim, c.embed_dim)
    def forward(self, x):
        # Extracting size to allow for variable length inputs
        N = x.size()[1]
        B = x.size()[0]
        # Splitting the obtained tensor and then treating it as separate objects of the same linear operation
        out = self.c_attn(x)

        q = torch.split(out, dim = 2, split_size_or_sections= c.embed_dim)[0].view(B, N, c.n_heads, c.embed_dim // c.n_heads).permute(0, 2, 1, 3)
        k = torch.split(out, dim = 2, split_size_or_sections= c.embed_dim)[1].view(B, N, c.n_heads, c.embed_dim // c.n_heads).permute(0, 2, 1, 3)
        v = torch.split(out, dim = 2, split_size_or_sections= c.embed_dim)[2].view(B, N, c.n_heads, c.embed_dim // c.n_heads).permute(0, 2, 1, 3)

        # Performing the attention mechanism
        attn_matrix = (q @ k.permute(0, 1, 3, 2))/torch.sqrt(torch.tensor(c.embed_dim//c.n_heads))
        mask = torch.triu(torch.ones(N, N, device=c.device), diagonal=1).bool()
        attn_matrix = attn_matrix.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        masked_attn = F.softmax(attn_matrix, dim = 3) @ v
        concat_attn = masked_attn.permute(0, 2, 1, 3).contiguous().view(B, N, c.embed_dim)
        # projecting the attention blocks
        out = self.c_proj(concat_attn)
        return out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(c.embed_dim, 4*c.embed_dim)
        self.c_proj = nn.Linear(4*c.embed_dim, c.embed_dim)

    def forward(self, x):
        out = self.c_fc(x)
        out = F.gelu(out)
        out = self.c_proj(out)
        return out


if __name__ == "__main__":
    model = GPT2()
    model.to(c.device)
    input = torch.ones([4, 7]).type("torch.IntTensor").to(c.device)
    out = model(input)
    print(out.size())