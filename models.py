import torch.nn.functional as F
import torch.nn as nn
import torch
from model_utils import get_model, TimmResNetWrapper

class def_model(torch.nn.Module):

    def __init__(self, resnet, classifier):

        super().__init__()
        self.resnet = resnet.resnet
        self.classifier = classifier

    def forward(self, x,return_feature=False):

        x = self.resnet.forward_features(x)
        x = self.resnet.global_pool(x)
        if self.resnet.drop_rate:
            x = torch.nn.functional.dropout(x, p=float(self.drop_rate), training=self.training)

        feature = x.view(x.size(0), -1)
        output = self.classifier(feature)
        if return_feature:
            return feature,output
        else:
            return output

class Classifier32(nn.Module):
    def __init__(self, num_classes=10, feat_dim=128):
        super(self.__class__, self).__init__()

        if feat_dim is None:
            feat_dim = 128

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    self.feat_dim,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(self.feat_dim)
        self.bn10 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(self.feat_dim, self.num_classes, bias=False)
        self.headfc = nn.Linear(self.feat_dim, self.feat_dim)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def encoder(self, x, normalize=False):
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)

        z = x.view(x.size(0), -1)

        if(normalize):
           z = F.normalize(z, dim=1)
        return z

        y = self.fc(x)
        if return_feature:
            return z, y
        else:
            return y

    def head(self,z,normalize = True):
        z = self.headfc(z)
        if(normalize):
           z = F.normalize(z, dim=1)
        return z

    def classifier(self,z,from_logits=True):
        z = self.fc(z)
        if(from_logits==False):
            z = torch.softmax(z, dim =1)
        return z


    def forward(self, x,normalize= True):
        z = self.encoder(x)
        output = self.head(z, normalize)
        #output = self.classifier(z)
        return output



class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim = 128, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.fc = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, features):
        return self.fc(features)


def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
def create_model(args):

    if args.architecture in ['timm_resnet50_pretrained','resnet50_pretrained','resnet50']:
       wrapper_class = TimmResNetWrapper
       model = get_model(args, wrapper_class=wrapper_class)
       if(args.encoder_lossname=='CE'):
           classifier = LinearClassifier(num_classes = args.N_closed,feat_dim=args.feat_dim)
           classifier = classifier.cuda()
           model=def_model(model,classifier)
           model = model.cuda()

    else:
       model = Classifier32(args.N_closed)   
       if(args.encoder_lossname=='CE'):
           classifier = LinearClassifier(num_classes = args.N_closed,feat_dim=args.feat_dim)
           classifier = classifier.cuda()
           model.headfc = classifier
           model = model.cuda()

    return model
