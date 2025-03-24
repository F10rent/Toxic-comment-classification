import torch
import torch.nn as nn

class MultiLabelEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, num_labels):
        super(MultiLabelEncoderClassifier, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embed_dim, num_labels)  # d_out: num_labels
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.emb(x)          # Embedding 
        x = self.encoder(x)      # Transformer 
        x = self.dropout(x)      # Dropout 
        x = x.max(dim=1)[0]      
        out = self.linear(x)     
        return torch.sigmoid(out)  



