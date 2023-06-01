# ©RoelDuijsings
from cProfile import label
import json
import os
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (get_choice, get_env, get_local_path,
                                   get_single_tag_keys, is_skipped)
from PIL import Image
import clip
import torch.nn as nn
import torch.optim as optim
import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

UNLABALED_DIR = r"C:\Users\roell\Documents\Master AI\Internship\Data\Unlabeled"
LABELED_DIR = r"C:\Users\roell\Documents\Master AI\Internship\Data\Labeled"

HOSTNAME = get_env('HOSTNAME', 'http://localhost:8080')
API_KEY = "003a52aa51e843ba009a78636dc3f6ca62023da4"
# API_KEY = get_env("KEY")

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
print('=> API_KEY = ', API_KEY)
if not API_KEY:
    print('=> WARNING! API_KEY is not set')

def convert_to_local_path(ls_path):
    # extract image name and join with unlabeled dir to get local image_path
    image_name= ls_path.split('-')[1]
    image_path = os.path.join(UNLABALED_DIR, image_name)
    return image_path



class CLIPDataset(Dataset):
    
    def __init__(self, image_paths, labels, preprocess):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess

        self.classes = list(set(labels))

    def __getitem__(self, index):
        image = self.preprocess(Image.open(self.image_paths[index]))
        label = clip.tokenize(self.labels[index])[0]
        return image, label

    def __len__(self):
        return len(self.image_paths)
    

class CLIPImageClassifier(object):
    """ 
    CLIP image classifier voor Label Studio Active Learning loop.
    ©RoelDuijsings
    """
    def __init__(self):#, T_max:int=0):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model = self.model.to(device)
        
        # Define a loss function for the images and texts
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        
        # Define an optimizer and a scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max) #T_max = len(dataloader)*self.num_epochs) #TODO: scheduler

        # return self.model, self.preprocess
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def predict(self, image_path, labels):
        # for path in image_paths:
        #     print(path)
        print("IMAGE:")
        print(image_path)
        print()
        # images = torch.stack([(self.preprocess(Image.open(path))) for path in image_path]).to(device)
        # labels = clip.tokenize(labels).to(device)
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(device) # TODO: now image per image, but can insert multiple images as a list to the model. The problem you're facing is related to the fact that your CLIP model is expecting a batch of images, but you're giving it a single image.In PyTorch, models expect inputs to have a batch dimension, even if there's only one item in the batch. In other words, if you're providing a single image, it still needs to be presented as a batch of size 1.
        print("Image shape:", image.shape)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, labels)
        print("Finished model prediction")
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
    

class CLIPImageClassifierAPI(LabelStudioMLBase):
    """ 
    CLIP image classifier API connectie voor Label Studio Active Learning loop.
    ©RoelDuijsings
    """
    def __init__(self, **kwargs):
        super(CLIPImageClassifierAPI, self).__init__( **kwargs)
        self.model = CLIPImageClassifier()
        
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'Choices', 'Image')
        
    def predict(self, tasks, **kwargs):
        # ls_image_paths = [task['data'][self.value] for task in tasks]
        # image_paths = [convert_to_local_path(ls_path) for ls_path in ls_image_paths]        # TODO: image paths to local image paths
        # print("Image paths:\n", image_paths)

        labels = self.classes
        labels = clip.tokenize(labels).to(device)

        predictions = []
        with torch.no_grad():
            for task in tasks:
                ls_image_path = task['data'][self.value]
                image_path = convert_to_local_path(ls_image_path)
                
                print("This image:", image_path)
                logits_per_image, logits_per_label = self.model.predict(image_path, labels)

                print("Calc pred..")
                probs = logits_per_image.softmax(dim=-1) #take the softmax to get the label probabilities
                k = min(5, len(self.classes)) # TODO: now only 2 classes
                top_probs, top_labels_indices = probs.topk(k=k, dim=-1) # returns values,indices

                topk_labels = [self.classes[id] for id in top_labels_indices[0]] # the top k predicted labels
                print(topk_labels)

                top_score = top_probs[0][0].item()
                predicted_label = topk_labels[0]

                result = [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'value': {'choices': [predicted_label]}
                }]
                
                # expand predictions with their scores for all tasks
                predictions.append({'result': result, 'score': float(top_score)})

                # uncertainty_score = calculate_uncertainty(top_probs)
                # list_uncertainty.append((path, true_label, topk_labels, top_probs, uncertainty_score))

        return predictions

    def fit(self, annotations, workdir=None, batch_size=10, num_epochs=20, **kwargs):
        image_paths, image_labels = [], []
        print('Collecting annotations...')
        
        # check if training is from webhook
        if kwargs.get('data'):
            project_id = kwargs['data']['project']['id']
            tasks = self._get_annotated_dataset(project_id)
            # print(f"tasks: {tasks}")
        # ML training without web hook
        else:
            tasks = annotations
        
        # extract image paths
        # TODO: nu worden ook niet geannoteerde images in de dataset toegevoegd, misschien niet doen?
        for task in tasks:
            # only add labeled images to dataset
            if not task.get('annotations'):
                continue
            annotation = task['annotations'][0] # get input text from task data
            if annotation.get('skipped') or annotation.get('was_cancelled'):
                continue
            
            # extract image name and join with unlabeled dir to get local image_path
            # image_name= task['data']['image'].split('-')[1]
            # image_path = os.path.join(UNLABALED_DIR, image_name )

            ls_path =  task['data']['image']
            image_path = convert_to_local_path(ls_path)
            image_paths.append(image_path)

            
            image_labels.append(annotation['result'][0]['value']['choices'][0])
        
        print()
        [print(img, label) for img, label in zip(image_paths ,image_labels)]
        # print(image_labels)
        print()



        print(f'Creating dataset with {len(image_paths)} images...')
        dataset = CLIPDataset(image_paths, image_labels, self.model.preprocess)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        print('Train model...')
        self.reset_model()
        self.model.train(dataloader, num_epochs=num_epochs)

        print('Save model...')
        model_path = os.path.join(workdir, 'model.pt')
        self.model.save(model_path)
        print("Finish saving.")

        return {'model_path': model_path, 'classes': dataset.classes}


    def _get_annotated_dataset(self, project_id):
        """Just for demo purposes: retrieve annotated data from Label Studio API"""
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'
        response = requests.get(download_url, headers={'Authorization': f'Token {API_KEY}'})
        if response.status_code != 200:
            raise Exception(f"Can't load task data using {download_url}, "
                            f"response status_code = {response.status_code}")
        return json.loads(response.content)
    
    def reset_model(self):
        self.model = CLIPImageClassifier()#len(self.classes), self.freeze_extractor)
