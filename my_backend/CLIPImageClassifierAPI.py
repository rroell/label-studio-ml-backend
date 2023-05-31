# ©RoelDuijsings
import json
import os
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from CLIPDataset import CLIPDataset
from CLIPImageClassifier import CLIPImageClassifier

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (get_choice, get_env, get_local_path,
                                   get_single_tag_keys, is_skipped)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


HOSTNAME = get_env('HOSTNAME', 'http://localhost:8080')
API_KEY = "003a52aa51e843ba009a78636dc3f6ca62023da4"
# API_KEY = get_env("KEY")

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
print('=> API_KEY = ', API_KEY)
if not API_KEY:
    print('=> WARNING! API_KEY is not set')


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
        image_paths = [task['data'][self.value] for task in tasks]
        labels = self.classes

        predictions = []
        for image_path in image_paths:
            logits_per_image, logits_per_label = self.model.predict(image_path, labels)

            probs = logits_per_image.softmax(dim=-1) #take the softmax to get the label probabilities
            top_probs, top_labels_indices = probs.topk(5, dim=-1) # returns values,indices

            topk_labels = [labels[id] for id in top_labels_indices[0]] # the top k predicted labels
            print(topk_labels)

            top_score = top_probs[0]
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

    def fit(self, annotations, workdir=None, batch_size=20, num_epochs=20, **kwargs):
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
            image_paths.append(task['data']['image'])
            if not task.get('annotations'):
                continue
            annotation = task['annotations'][0] # get input text from task data
            if annotation.get('skipped') or annotation.get('was_cancelled'):
                continue
            
            image_labels.append(annotation['result'][0]['value']['choices'][0])
        
        print(f'Creating dataset with {len(image_paths)} images...')
        dataset = CLIPDataset(image_paths, image_labels)
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
        self.model = CLIPImageClassifier(len(self.classes), self.freeze_extractor)
