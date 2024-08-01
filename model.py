import torch
from torch import nn
from transformers import BertModel, XLNetForTokenClassification

def freeze_bert_weights(model:BertModel, freeze_layer:int=12, freeze_embedding:bool=True, freeze_pool:bool=True):
    for i in range(0, freeze_layer):
        for param in model.encoder.layer[i].parameters():
            param.requires_grad = False
        print(f'froze encoder layer {i}')
        

    if freeze_embedding:
        for param in model.embeddings.parameters():
            param.requires_grad = False
        print('froze embedding')

    if freeze_pool:
        for param in model.pooler.parameters():
            param.requires_grad = False 
        print('froze pooler')

def freeze_xlnet_weights(model:XLNetForTokenClassification, freeze_layer:int=12, freeze_embedding:bool=True, freeze_summary:bool=True):
    for i in range(0, freeze_layer):
        for param in model.transformer.layer[i].parameters():
            param.requires_grad = False
        print(f'froze transformer layer {i}')

    if freeze_embedding:
        for param in model.transformer.word_embedding.parameters():
            param.requires_grad = False
        print('froze embedding')

    if freeze_summary:
        for param in model.sequence_summary.parameters():
            param.requires_grad = False 
        print('froze summary')

class TransformerClassifier(nn.Module):
    def __init__(self, backbone, input_size:int=768, hidden_layers:list[int]=[], num_classes:int=8):
        super(TransformerClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = nn.ModuleList()

        cls_layers = [input_size] + hidden_layers
        for i in range(len(cls_layers) - 1):
            self.classifier.append(nn.Sequential(
                nn.Linear(cls_layers[i], cls_layers[i + 1]),
                nn.ReLU()
            ))
        self.classifier.append(nn.Linear(cls_layers[-1], num_classes))

    def forward(self, **args):
        outputs = self.backbone(**args)
        out = outputs.pooler_output
        for module in self.classifier:
            out = module(out)
        return out


if __name__ == '__main__':
    from transformers import BertModel
    from transformers import XLNetForSequenceClassification
    
    # backbone = BertModel.from_pretrained('bert-base-uncased')
    # backbone = BertModel.from_pretrained('roberta-base')
    backbone = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased') 
    print(backbone)

    freeze_xlnet_weights(backbone, freeze_layer=1, freeze_embedding=True, freeze_summary=True)
    # freeze_bert_weights(backbone, freeze_layer=4, freeze_embedding=True, freeze_pool=True)
    
    model = TransformerClassifier(backbone, 'xlnet', hidden_layers=[128])
    print(model)
    for name, param in model.named_parameters():
        # Layer numbers are part of parameter names. E.g., 'encoder.layer.0.attention.self.query.weight'
        print(name, param.requires_grad)