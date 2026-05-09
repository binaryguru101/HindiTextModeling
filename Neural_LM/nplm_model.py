import torch
import torch.nn as nn

class NPLM(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=128, context_size=3, hidden_dim=512, use_second_hidden=False):
        super(NPLM, self).__init__()

        # lookup table
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.use_second_hidden = use_second_hidden
        
        # hidden layer with tanh (as the non-linear) activation
        self.hidden_layer = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.tanh = nn.Tanh()
        
        # optional second hidden layer for architecture 2
        if self.use_second_hidden:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.act2 = nn.Tanh()

        # output projection
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embeddings(x)

        x_concat = embeds.view(embeds.size(0), -1)
        
        h = self.tanh(self.hidden_layer(x_concat))

        if self.use_second_hidden:
            h = self.act2(self.fc2(h))

        y = self.output_layer(h)
        
        return y

if __name__ == "__main__":
    V_SIZE, E_DIM, C_SIZE, H_DIM = 10000, 128, 3, 512
    model = NPLM(V_SIZE, E_DIM, C_SIZE, H_DIM)
    dummy = torch.randint(0, V_SIZE, (2, C_SIZE))
    output = model(dummy)
    print("\n--- DEBUG ---")
    print(model)
    print(f"Output Shape: {output.shape}")

    V_SIZE, E_DIM, C_SIZE, H_DIM = 10000, 128, 3, 512
    model = NPLM(V_SIZE, E_DIM, C_SIZE, H_DIM, use_second_hidden=True)
    dummy = torch.randint(0, V_SIZE, (2, C_SIZE))
    output = model(dummy)
    print("\n--- DEBUG use_second_hidden = True ---")
    print(model)
    print(f"Output Shape: {output.shape}")