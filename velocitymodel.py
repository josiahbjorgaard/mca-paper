import torch
from torch import nn
from torchmetrics.functional.regression import pearson_corrcoef


## Start model training and defining the training loop
class VelocityModel(nn.Module):
    def __init__(self, backbone_model, output_size, backbone_hidden_size, decoder_hidden_size = 256, layer_to_unfreeze=None,decoder_num_layers=3, dropout=0.1, tokens_to_fit=None):
        super().__init__()
        for name,param in backbone_model.named_parameters():
            if not layer_to_unfreeze or layer_to_unfreeze not in name:
                print(name)
                param.requires_grad=False
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


