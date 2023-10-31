import torch
from torch import nn
from torchmetrics.functional.regression import pearson_corrcoef
from encoders import GeneEncoder
from multizorromodel import TokenTypes, Attention

## Start model training and defining the training loop
class VelocityModel(nn.Module):
    def __init__(self, backbone_model, output_size, backbone_hidden_size, decoder_hidden_size = 256, layers_to_unfreeze=None,decoder_num_layers=3, dropout=0.1, tokens_to_fit=None):
        super().__init__()
        if layers_to_unfreeze != "all":
            for name,param in backbone_model.named_parameters():
                param.requires_grad = True if name in layers_to_unfreeze else False
                print(f"{name}:{param.requires_grad}")
        self.backbone_model = backbone_model
        self.dropout = nn.Dropout(dropout)
        if decoder_num_layers == 0:
            decoder_hidden_size = output_size
        layers = [
                    nn.Linear(backbone_hidden_size, decoder_hidden_size),
                 ]
        for n in range(decoder_num_layers):
            if n != decoder_num_layers-1:
                decoder_hidden_size_in = decoder_hidden_size
                decoder_hidden_size_out = decoder_hidden_size
            else:
                decoder_hidden_size_in = decoder_hidden_size
                decoder_hidden_size_out = output_size
            layers = layers + [
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(decoder_hidden_size_in, decoder_hidden_size_out),
                              ]
        self.decoder = nn.Sequential(*layers)
        print(self.decoder)
        self.loss = nn.MSELoss()
        self.metrics = {"pcc":pearson_corrcoef}
    def forward(self, batch, return_logit=False):
        output = self.backbone_model(**{k:v for k,v in batch.items() if 'velocity' not in k}, no_loss=True)
        logits = torch.cat([output.expression, output.spliced, output.unspliced, output.fusion], dim=1)
        logits = self.decoder(logits)
        loss, metric = self.loss(logits, batch['velocity']), {k:metric(logits.flatten(), batch['velocity'].flatten()) for k,metric in self.metrics.items()}
        if return_logit:
            return loss, metric, logits
        else:
            return loss, metric


class VelocityHiddenModel(nn.Module):
    """
    Get the final hidden state (before pooling), and initialize a new Cross-Attention 'pooling' layer to make queries with their own nn.Embedding
    """
    def __init__(self,
                 backbone_model,
                 output_size,
                 backbone_hidden_size = 256,
                 output_token_type = TokenTypes.GLOBAL,
                 layers_to_unfreeze=None,
                 dim_head = 64,
                 heads = 8,
                 decoder_num_layers=3,
                 dropout=0.1,
                 vocab_size=2000):
        super().__init__()

        # Backbone Model
        if layers_to_unfreeze != "all":
            for name,param in backbone_model.named_parameters():
                param.requires_grad = True if name in layers_to_unfreeze else False
                #print(f"{name}:{param.requires_grad}")
        self.backbone_model = backbone_model

        # Pooling Cross-Attention Layer
        return_token_types = (output_token_type, ) * output_size #N.B. This could be a mix of token types, like 2000 * EXPRESSION, 2000 * GLOBAL
        self.embedding = GeneEncoder(vocab_size, backbone_hidden_size, padding_idx=0, max_norm = 1.0)
        self.max_return_tokens = len(return_token_types)
        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(list(map(lambda t: t.value, return_token_types)))
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)
        self.attn_pool = Attention(dim=backbone_hidden_size, dim_head=dim_head, heads=heads)
        # End Pooling Cross-Attention Layer

        # MLP Decoder for Regression
        self.dropout = nn.Dropout(dropout)
        if decoder_num_layers == 0:
            decoder_hidden_size = output_size
        layers = [
                    nn.Linear(backbone_hidden_size, decoder_hidden_size),
                 ]
        for n in range(decoder_num_layers):
            if n != decoder_num_layers-1:
                decoder_hidden_size_in = decoder_hidden_size
                decoder_hidden_size_out = decoder_hidden_size
            else:
                decoder_hidden_size_in = decoder_hidden_size
                decoder_hidden_size_out = output_size
            layers = layers + [
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(decoder_hidden_size_in, decoder_hidden_size_out),
                              ]
        self.decoder = nn.Sequential(*layers)

        # Loss and Metrics
        self.loss = nn.MSELoss()
        self.metrics = {"pcc":pearson_corrcoef}
    def forward(self, batch, return_logit=False):
        final_hidden_state, token_types_attend_to = self.backbone_model(**{k:v for k,v in batch.items() if 'velocity' not in k}, return_final_hidden_state = True)

        #Function for cross-attention pooling layer
        #return_tokens = repeat(return_tokens, 'n d -> b n d', b=batch)
        return_tokens = self.embedding(batch) ###!! Add the actual token numbers here)) -> b n d
        return_token_types_tensor = self.return_token_types_tensor

        #  self attention for non-global tokens
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to
        # global queries can attend to all tokens
        pool_mask = pool_mask | (rearrange(return_token_types_tensor, 'i -> i 1') == torch.ones_like(
            token_types_attend_to, dtype=torch.long) * TokenTypes.GLOBAL.value)

        # Padding mask to pool mask
        padding_mask = repeat(padding, 'b j -> b i j', i=pool_mask.shape[0])
        pool_mask = pool_mask * padding_mask
        pool_mask = repeat(pool_mask, 'b i j -> b h i j', h=self.heads)

        pooled_tokens = self.attn_pool(return_tokens, context=final_hidden_state, attn_mask=pool_mask) + return_tokens

        logits = torch.cat([output.expression, output.spliced, output.unspliced, output.fusion], dim=1)
        logits = self.decoder(logits)
        loss, metric = self.loss(logits, batch['velocity']), {k:metric(logits.flatten(), batch['velocity'].flatten()) for k,metric in self.metrics.items()}
        if return_logit:
            return loss, metric, logits
        else:
            return loss, metric