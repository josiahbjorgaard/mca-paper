import torch
from torch import nn
from torchmetrics.functional.regression import pearson_corrcoef
from encoders import GeneEncoder
from multizorromodel import TokenTypes, Attention, BioZorroLayer
from einops import rearrange, repeat, pack, unpack

def sample_min(pred,target):
    return torch.min(target)

def sample_max(pred,target):
    return torch.max(target)

def sample_mean(pred,target):
    return torch.mean(target)

## Start model training and defining the training loop
class VelocityModel(nn.Module):
    def __init__(self, backbone_model, output_size, backbone_hidden_size, decoder_hidden_size = 256, layers_to_unfreeze=None,decoder_num_layers=3, dropout=0.1, tokens_to_fit=None, output_types=["fusion","global_output"]):
        super().__init__()
        if layers_to_unfreeze != "all":
            for name,param in backbone_model.named_parameters():
                param.requires_grad = False
                if layers_to_unfreeze and name in layers_to_unfreeze:
                    param.requires_grad = True
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
        print(f"{self.decoder = }")
        self.loss = nn.MSELoss()
        self.metrics = {"pcc":pearson_corrcoef, "min":sample_min, "max": sample_max, "mean": sample_mean}
        self.output_types = output_types #"expression","spliced","unspliced","fusion","global_output"
    def forward(self, batch, return_logit=False):
        output = self.backbone_model(**{k:v for k,v in batch.items() if 'velocity' not in k}, no_loss=True)
        logits = torch.cat([output[otype] for otype in self.output_types], dim=1)
        logits = self.decoder(logits)
        loss, metric = self.loss(logits, batch['velocity']), {k:metric(logits.flatten(), batch['velocity'].flatten()) for k,metric in self.metrics.items()}
        if return_logit:
            return loss, metric, logits
        else:
            return loss, metric


class VelocityHiddenModel(nn.Module):
    """
    Get the final hidden state (before pooling), and initialize a new Cross-Attention 'pooling' layer
    to make queries with their own GeneEmbedding
    """
    def __init__(self,
                 backbone_model,
                 output_size,
                 backbone_hidden_size = 256,
                 output_token_type = TokenTypes.GLOBAL,
                 layers_to_unfreeze=None,
                 dim_head = 64,
                 heads = 8,
                 attn_layers=0,
                 ff_mult=4,
                 decoder_num_layers=3,
                 decoder_hidden_size=256,
                 dropout=0.1,
                 vocab_size=2000):
        super().__init__()

        # Backbone Model
        if layers_to_unfreeze != "all":
            for name,param in backbone_model.named_parameters():
                param.requires_grad = False
                if layers_to_unfreeze and name in layers_to_unfreeze:
                    print(f"unfreezing {name}")
                    param.requires_grad = True
        self.backbone_model = backbone_model

        # Pooling Cross-Attention Layer
        return_token_types = (output_token_type, ) * output_size #N.B. This could be a mix of token types, like 2000 * EXPRESSION, 2000 * GLOBAL
        self.embedding = GeneEncoder(vocab_size, backbone_hidden_size, padding_idx=0, max_norm = 1.0)
        self.max_return_tokens = len(return_token_types)
        self.return_token_types = return_token_types
        return_token_types_tensor = torch.tensor(list(map(lambda t: t.value, return_token_types)))
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)
        self.attn_pool = Attention(dim=backbone_hidden_size, dim_head=dim_head, heads=heads)
        self.heads = heads
        # End Pooling Cross-Attention Layer

        #Additional Transformer Layers TODO: add paddding mask
        layers = []
        for _ in range(attn_layers):
            layers.append(BioZorroLayer(dim=backbone_hidden_size, dim_head=dim_head, heads=heads, ff_mult=ff_mult)) 
        self.attn_layers = nn.Sequential(*layers)

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
                decoder_hidden_size_out = 1 #output_size
            layers = layers + [
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(decoder_hidden_size_in, decoder_hidden_size_out),
                              ]
        self.decoder = nn.Sequential(*layers)

        # Loss and Metrics
        self.loss = nn.MSELoss()
        self.metrics = {"pcc":pearson_corrcoef,"max":sample_max,"min":sample_min, "mean",sample_mean}

    def forward(self, batch, return_logit=False):
        final_hidden_state, \
            token_types_attend_to, \
            padding = self.backbone_model(**{k: v for k, v in batch.items() if 'velocity' not in k},
                                          return_final_hidden_state=True, no_loss=True)

        #Function for cross-attention pooling layer
        #return_tokens = repeat(return_tokens, 'n d -> b n d', b=batch)
        return_tokens = self.embedding(batch['velocity_index']) ###!! Add the actual token numbers here)) -> b n d
        return_token_types_tensor = self.return_token_types_tensor
        #print(f"return_tokens: {return_tokens.shape}")

        #  self attention for non-global tokens
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to
        # global queries can attend to all tokens
        pool_mask = pool_mask | (rearrange(return_token_types_tensor, 'i -> i 1') == torch.ones_like(
            token_types_attend_to, dtype=torch.long) * TokenTypes.GLOBAL.value)
        #print(f"pool_mask: {pool_mask.shape}")

        # Padding mask to pool mask
        padding_mask = repeat(padding, 'b j -> b i j', i=pool_mask.shape[0])
        #print(f"pad_mask: {padding_mask.shape}")

        pool_mask = pool_mask * padding_mask
        pool_mask = repeat(pool_mask, 'b i j -> b h i j', h=self.heads)

        #print(f"pool_mask*pad_mask reapeasted: {pool_mask.shape}")
        tokens = self.attn_pool(return_tokens, context=final_hidden_state, attn_mask=pool_mask) + return_tokens
        #print(f"{tokens.shape = }")
        
        tokens = self.attn_layers(tokens)

        #Regression Decoder
        logits = self.decoder(tokens).squeeze()
        #print(f"{logits.shape = }")
        loss_mask = batch['velocity_index'] != 0
        targets = batch['velocity_data'][loss_mask]
        logits = logits[loss_mask]
        loss = self.loss(logits, targets)
        metric =  { k: metric(logits.flatten(), targets.flatten()) for k, metric in self.metrics.items() }
        if return_logit:
            return loss, metric, logits
        else:
            return loss, metric
