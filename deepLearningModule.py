import matplotlib
from torch._C import NoneType

# General utilities
import os
import glob
import time
import subprocess
import scipy.io
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm   
from sklearn import preprocessing

# Torch and Torchvision
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchsummary import summary
import torch.nn as nn

# Visualisation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
print("CUDA available: ", torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Custom Flower Dataset Class

class FlowerDataset(Dataset):
    def __init__(self, image_directory, annotation_file, transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotation_file)
        self.image_directory = image_directory
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 0]
        image = Image.open(os.path.join(self.image_directory, img_path)).convert("RGB")
        label = self.annotations.iloc[index, 1]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

class DeepLearningEnviroment:
  label_to_species_dict ={
                        '21': 'fire lily',
                        '3': 'canterbury bells',
                        '45': 'bolero deep blue',
                        '1': 'pink primrose',
                        '34': 'mexican aster',
                        '27': 'prince of wales feathers',
                        '7': 'moon orchid',
                        '16': 'globe-flower',
                        '25': 'grape hyacinth',
                        '26': 'corn poppy',
                        '79': 'toad lily',
                        '39': 'siam tulip',
                        '24': 'red ginger',
                        '67': 'spring crocus',
                        '35': 'alpine sea holly',
                        '32': 'garden phlox',
                        '10': 'globe thistle',
                        '6': 'tiger lily',
                        '93': 'ball moss',
                        '33': 'love in the mist',
                        '9': 'monkshood',
                        '102': 'blackberry lily',
                        '14': 'spear thistle',
                        '19': 'balloon flower',
                        '100': 'blanket flower',
                        '13': 'king protea',
                        '49': 'oxeye daisy',
                        '15': 'yellow iris',
                        '61': 'cautleya spicata',
                        '31': 'carnation',
                        '64': 'silverbush',
                        '68': 'bearded iris',
                        '63': 'black-eyed susan',
                        '69': 'windflower',
                        '62': 'japanese anemone',
                        '20': 'giant white arum lily',
                        '38': 'great masterwort',
                        '4': 'sweet pea',
                        '86': 'tree mallow',
                        '101': 'trumpet creeper',
                        '42': 'daffodil',
                        '22': 'pincushion flower',
                        '2': 'hard-leaved pocket orchid',
                        '54': 'sunflower',
                        '66': 'osteospermum',
                        '70': 'tree poppy',
                        '85': 'desert-rose',
                        '99': 'bromelia',
                        '87': 'magnolia',
                        '5': 'english marigold',
                        '92': 'bee balm',
                        '28': 'stemless gentian',
                        '97': 'mallow',
                        '57': 'gaura',
                        '40': 'lenten rose',
                        '47': 'marigold',
                        '59': 'orange dahlia',
                        '48': 'buttercup',
                        '55': 'pelargonium',
                        '36': 'ruby-lipped cattleya',
                        '91': 'hippeastrum',
                        '29': 'artichoke',
                        '71': 'gazania',
                        '90': 'canna lily',
                        '18': 'peruvian lily',
                        '98': 'mexican petunia',
                        '8': 'bird of paradise',
                        '30': 'sweet william',
                        '17': 'purple coneflower',
                        '52': 'wild pansy',
                        '84': 'columbine',
                        '12': "colt's foot",
                        '11': 'snapdragon',
                        '96': 'camellia',
                        '23': 'fritillary',
                        '50': 'common dandelion',
                        '44': 'poinsettia',
                        '53': 'primula',
                        '72': 'azalea',
                        '65': 'californian poppy',
                        '80': 'anthurium',
                        '76': 'morning glory',
                        '37': 'cape flower',
                        '56': 'bishop of llandaff',
                        '60': 'pink-yellow dahlia',
                        '82': 'clematis',
                        '58': 'geranium',
                        '75': 'thorn apple',
                        '41': 'barbeton daisy',
                        '95': 'bougainvillea',
                        '43': 'sword lily',
                        '83': 'hibiscus',
                        '78': 'lotus lotus',
                        '88': 'cyclamen',
                        '94': 'foxglove',
                        '81': 'frangipani',
                        '74': 'rose',
                        '89': 'watercress',
                        '73': 'water lily',
                        '46': 'wallflower',
                        '77': 'passion flower',
                        '51': 'petunia'}

  le = preprocessing.LabelEncoder()

  def getSpeciesDictionary(self):
    return self.label_to_species_dict

  def downloadFlowerData(self):
    subprocess.run('wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz', shell=True)
    subprocess.run('wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat', shell=True)
    subprocess.run("mkdir /content/flowers", shell=True)
    subprocess.run("tar -zxvf 102flowers.tgz -C /content/flowers", shell=True)

  def downloadLeafData(self):
    subprocess.run('wget -O leaves_no_augmentation.zip https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded', shell=True)
    subprocess.run("mkdir /content/leaves", shell=True)
    subprocess.run("unzip leaves_no_augmentation -d /content/leaves", shell=True)

  def createLeafDataset(self, trainSetFraction=0.8, seed=42):

    # Create a dataframe for selected species of plants

    ## Parameters
    species_name = 'Grape'
    images_per_label = 250

    ## Loop through the folder and save the image name + labels into Python lists
    img_paths = []
    labels = []
    for folder_name in glob.glob('/content/leaves/Plant_leave_diseases_dataset_without_augmentation/' + species_name + '*'):
      label = folder_name.rsplit('___', 1)[1].title().replace('_', ' ')
      for img in glob.glob(folder_name+'/*'):
        img_paths.append(img)
        labels.append(label)

    ## Create and sample Pandas dataframe from the selected species
    annotation_frame = pd.DataFrame({'img_name': img_paths, 'label': labels})
    annotation_frame = annotation_frame.groupby('label', group_keys=False).apply(lambda x: x.sample(images_per_label)).reset_index(drop=True)

    ## Verbose the label distribution
    print('Label distribution:\n')
    print(annotation_frame['label'].value_counts())

    # Label encding for the initial training
    annotation_frame['label'] = self.le.fit_transform(annotation_frame['label'])

    # Split dataframe into training and validation

    ## Parameters
    train_csv_name = 'trainPlantDisease'
    validation_csv_name = 'validationPlantDisease'

    ## Custom Stratified train/validation split
    train_df = annotation_frame.groupby('label', as_index=False).apply(lambda x: x.sample(frac=trainSetFraction, random_state=seed)).reset_index()[['img_name', 'label']]
    validation_df = annotation_frame[~annotation_frame.img_name.isin(train_df.img_name)]

    ## Write to CSV files - Verbose dataset shapes
    train_df.to_csv (train_csv_name + '.csv', index = False, header=True)
    validation_df.to_csv (validation_csv_name + '.csv', index = False, header=True)

    ## Verbose
    print('Train Set:'.ljust(15), train_df.shape)
    print('Validation Set:'.ljust(15), validation_df.shape)

  def createFlowerDataset(self, labels=None, trainSetFraction=0.8, seed=42):
    # Store the image paths and labels in a Pandas dataframe
    if labels is None:
      labels = list(range(1, 11))

    ## The labels are originally Matlab data, we use SciPy to load it into a list
    label_mat = scipy.io.loadmat('imagelabels.mat')

    ## Create dataframe: Column 1 is image paths, Column 2 is labels
    image_folder = '/content/flowers/jpg'
    annotation_frame = pd.DataFrame({'img_name': sorted([img for img in os.listdir(image_folder)]),
                                     'label': label_mat['labels'][0]}) 
    
    # Adjust dataframes for initial training and randomly selected species
    random_annotation_frame = annotation_frame[annotation_frame['label'].isin(labels)]
    annotation_frame = annotation_frame[~annotation_frame['label'].isin(labels)]

    #print('Initial annotation dataframe [100 species]: '.ljust(45), annotation_frame.shape)
    print('Chosen flower species dataframe: '.ljust(38), random_annotation_frame.shape)

    # Label encding for the initial training
    random_annotation_frame['label'] = self.le.fit_transform(random_annotation_frame['label'])

    # Split dataframe into training and validation

    ## Parameters
    train_csv_name = 'trainFlowers'
    validation_csv_name = 'validationFlowers'

    ## Custom Stratified train/validation split
    train_df = random_annotation_frame.groupby('label', as_index=False).apply(lambda x: x.sample(frac=trainSetFraction, random_state=seed)).reset_index()[['img_name', 'label']]
    validation_df = random_annotation_frame[~random_annotation_frame.img_name.isin(train_df.img_name)]

    ## Write to CSV files - Verbose dataset shapes
    train_df.to_csv (train_csv_name + '.csv', index = False, header=True)
    validation_df.to_csv (validation_csv_name + '.csv', index = False, header=True)

    ## Verbose
    print('Train Set:'.ljust(38), train_df.shape)
    print('Validation Set:'.ljust(38), validation_df.shape)
    print('Train and validation datasets are saved as ' + train_csv_name + '.csv' + ' and ' + validation_csv_name + '.csv' + ' respectively.')

  def getDataLoader(self, imageFolder, annotationFile, transformation, batchSize):
    dataset = FlowerDataset(imageFolder, annotationFile, transform=transformation)
    dataLoader = DataLoader(dataset=dataset, shuffle=True, batch_size=batchSize, num_workers=2, pin_memory=True)
    return dataLoader

  def createModel(self, pretrained, nClasses):
    
    # Define model
    if pretrained:
      model = models.vgg16(weights='VGG16_Weights.DEFAULT')
      model.classifier[6] = nn.Linear(4096, nClasses)
    else:
      model = models.vgg16(weights=None)
      model.classifier = nn.Sequential(nn.Linear(25088, 4096, bias = True),
                                            nn.ReLU(inplace = True),
                                            nn.Dropout(0.4),
                                            nn.Linear(4096, 2048, bias = True),
                                            nn.ReLU(inplace = True),
                                            nn.Dropout(0.4),
                                            nn.Linear(2048, nClasses))
    
    # Get device and load model to the device
    torch.cuda.empty_cache()
    model.to(device)
    print('VGG16 model created successfully, summary; \n', model)
    return model
    
  def trainModel(self, model, trainDataLoader, validationDataLoader, learningRate, epochs):
    # Training parameters 
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
    criterion = nn.CrossEntropyLoss()

    ## Define list objects for storing training information 
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for data, label in tqdm(trainDataLoader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(trainDataLoader)
            epoch_loss += loss / len(trainDataLoader)
        train_accuracy.append(epoch_accuracy.item())
        train_loss.append(epoch_loss.item())
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in validationDataLoader:
                data = data.to(device)
                label = label.to(device)
                val_output = model(data)
                val_loss = criterion(val_output, label)
                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(validationDataLoader)
                epoch_val_loss += val_loss / len(validationDataLoader)
            valid_accuracy.append(epoch_val_accuracy.item())
            valid_loss.append(epoch_val_loss.item())
        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
    return train_loss, train_accuracy, valid_loss, valid_accuracy

  def getPredictionsSingleImage(self, model, imagePath, transform=None):
    model.eval()
    image = Image.open(imagePath).convert("RGB")
    matplotlib.pyplot.imshow(image)
    plt.show()
    if transform is not None:
      image = transform(image)
    with torch.no_grad():
      predictions = model(image.unsqueeze(dim=0).to(device))
      pred_labels = torch.softmax(predictions, 1).argmax()
      print('Prediction: ', self.label_to_species_dict[str(self.le.inverse_transform([pred_labels.item()])[0])])
      print('Probability: ', round(torch.softmax(predictions, 1).max().item(), 4))

  def plot_model_history(self, history, model_name, loss_type):

    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)  

    # Subplots (Vertically stacked)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(model_name + ' Training Metrics Plot', fontsize=16, y=1.05)

    ax1.spines["top"].set_visible(False)    
    ax1.spines["bottom"].set_visible(False)    
    ax1.spines["right"].set_visible(False)    
    ax1.spines["left"].set_visible(False) 

    ax1.plot(np.arange(1, len(history[1])+1, 1.0), np.array(history[1])*100, color=tableau20[0])
    ax1.plot(np.arange(1, len(history[1])+1, 1.0), np.array(history[3])*100, color=tableau20[12])
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(np.arange(1, len(history[3])+1, 1.0))
    ax1.set_yticks(np.arange(0, 100+0.05, 5))

    ax2.spines["top"].set_visible(False)    
    ax2.spines["bottom"].set_visible(False)    
    ax2.spines["right"].set_visible(False)    
    ax2.spines["left"].set_visible(False) 

    ax2.plot(np.arange(1, len(history[0])+1, 1.0), history[0], color=tableau20[0])
    ax2.plot(np.arange(1, len(history[0])+1, 1.0), history[2], color=tableau20[12])
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(loss_type)
    ax2.set_xticks(np.arange(1, len(history[0])+1, 1.0))
    ax2.set_yticks(np.arange(0, max(history[2])+0.05, 0.2))
    ax2.legend(['Train', 'Validation'], loc='upper right')