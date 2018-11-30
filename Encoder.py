import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    """ Encoder part -- A CNN Encoder to produce the features of given image """
    def __init__(self, base_model="inception", init=True, feature_size=2048, train=False):
        """
        Args:
            base_model (string) - Default: "inception" 
                name of the CNN model we would like to use. Should be one of the:
                {"vgg16", "vgg19", "resnet50", "resnet101", "resnet152", "inception"}
            init (bool) - Default: True
                Whether use the pre-trained based model to initialize parameters
            feature_size (int) - Default: 2048 
                size of the output feature from CNN. Should match the hidden
                state size in the RNN Decoder.
            train (bool) - Default: False
                In Phase 1 we should freeze the CNN parameters (other than the last 
                feature layer). In Phase 2 we will fine tune all parameters
    
        Returns:
            None
        """

        super(Encoder, self).__init__()

        """ Load the pretrianed model """
        self.model = None
        if base_model == "vgg16" :
            self.model = models.vgg16_bn(init)
        if base_model == "vgg19" :
            self.model = models.vgg19_bn(init)
        if base_model == "resnet50" :
            self.model = models.resnet50(init)
        if base_model == "resnet101" :
            self.model = models.resnet101(init)
        if base_model == "resnet152" :
            self.model = models.resnet152(init)
        if base_model == "inception":
            self.model = models.inception_v3(init)
        assert self.model is not None, "Error, unknown CNN model structure"
        
        """ Replace the last FC layer/classifier """
        if base_model.startswith("resnet"):
            self.fc     = nn.Linear(self.model.fc.in_features, feature_size)
            modules     = list(self.model.children())[:-1]
            self.model  = nn.Sequential(*modules)
            self.bn     = nn.BatchNorm1d(feature_size, momentum=0.01)
        
        if base_model.startswith("vgg"):
            self.fc     = nn.Linear(self.model.classifier[0].in_features, feature_size)
            modules     = list(self.model.children())[:-1]
            self.model  = nn.Sequential(*modules)
            self.bn     = nn.BatchNorm1d(feature_size, momentum=0.01)

        if base_model.startswith("inception"):
            self.fc     = nn.Linear(self.model.fc.in_features, feature_size)
            modules     = list(self.model.children())[:-1]
            self.model  = nn.Sequential(*modules)
            self.bn     = nn.BatchNorm1d(feature_size, momentum=0.01)
        
        """ Freeze the CNN part in the 1st Phase """
        self.fine_tune(train)

        """ Initialize the weights in the FC layer """
        if init:
            self.fc.weight.data.normal_(0.0, 0.02)
            self.fc.bias.data.fill_(0)

    def fine_tune(self, allow_fine_tune=True):
        """ Freeze the CNN part in the 1st Phase """
        for param in self.model.parameters():
            param.requires_grad = allow_fine_tune

    def forward(self, images):
        """ 
        Args:
            images: (Tensor) Batch of input images

        Returns:
            features: (Tensor) The extracted features from the Encoder part
        """
        
        features = self.model(images)
        features = features.reshape(features.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)

        return features