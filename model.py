import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
                                ##(512 , 256 , 9995)
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()

        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,dropout=0.25,batch_first = True)
        self.fc = nn.Linear(hidden_size,vocab_size)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.hidden_size = hidden_size;

        return
    
    def forward(self, features, captions):
        pred = features.unsqueeze(1)
        capt = self.embedding(captions[:,:-1])       ## last token is generated but not used for input again
                                                     ## so we remove it from input sequence
        
        catted = torch.cat((pred,capt),dim=1)        ## input sequence starts with image features , and ends with
        pred,hidden = self.lstm(catted)
    
        pred = pred.contiguous().view(-1,self.hidden_size)
        pred = self.fc(pred);
        
        return pred
        
                       
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        out = []
        
        pred,hid = self.lstm(inputs,states)
        word = self.fc(pred.squeeze())
        word = torch.argmax(word,-1)
        out.append(word)
        for i in range(max_len-1):
            inp = self.embedding(out[-1])
#            print('inp.shape = ',inp.shape)
#            print('hid[0].shape = ',hid[0].shape)
#            print('hid[1].shape = ',hid[1].shape)
            out_,hid = self.lstm(inp.view(-1,1,self.hidden_size),hid)
            word = self.fc(out_.squeeze())
            word = torch.argmax(word,-1)
            out.append(word)
            
        out = [o.item() for o in out]
        return out
