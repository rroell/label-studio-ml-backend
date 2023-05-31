from email.mime import image
import torch
import torch.nn as nn
import torch.optim as optim
import time
import clip
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CLIPImageClassifier(object):
    """ 
    CLIP image classifier voor Label Studio Active Learning loop.
    ©RoelDuijsings
    """
    def __init__(self, T_max:int=0):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model = self.model.to(device)
        
        # Define a loss function for the images and texts
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        
        # Define an optimizer and a scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max) #T_max = len(dataloader)*self.num_epochs)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def predict(self, image_paths, labels):
        images = torch.stack([self.preprocess(Image.open(path)) for path in image_paths]).to(device)
        labels = clip.tokenize(labels).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(images, labels)

        return logits_per_image.to(device), logits_per_text.to(device)
    
    def train(self, dataloader, num_epochs=20):
        since = time.time()

        #https://github.com/openai/CLIP/issues/57
        def convert_models_to_fp32(model): 
            for p in model.parameters(): 
                p.data = p.data.float() 
                p.grad.data = p.grad.data.float() 

        self.model.train()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            
            running_loss = 0.0

            # Iterate over data.
            for batch in dataloader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                self.optimizer.zero_grad()
                logits_per_image, logits_per_text = self.model(images, labels)
                ground_truth = torch.arange(len(images), device=device)

                total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text,ground_truth)) / 2
                total_loss.backward()

                if device == "cpu":
                    self.optimizer.step()
                else : 
                    convert_models_to_fp32(self.model)
                    self.optimizer.step()
                    clip.model.convert_weights(self.model)

                # loss statistics
                running_loss += total_loss.item() * images.size(0)
                # self.scheduler.step()

            # loss per epoch
            epoch_loss = running_loss / len(dataloader.dataset)
            print('Train Loss: {:.4f}'.format(epoch_loss))

        print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return self.model