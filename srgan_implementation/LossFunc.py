from torch import nn 
from torchvision.models import vgg19, VGG19_Weights
import config as config 

#phi_5,4 5th conv layer before maxpooling but after activation

class VGGLoss(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters(): 
            param.requires_grad = False
        
    def forward(self, input, target): 
        
        # Pass input and target through VGG
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
    
        
        # Compute MSE loss
        loss_value = self.loss(vgg_input_features, vgg_target_features)
        
        return loss_value