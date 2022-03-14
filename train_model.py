from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
import os
import torch
import torch.nn.functional as F
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters

from torchmeta.transforms import ClassSplitter
import numpy as np
import random
import pickle
import copy

from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import transforms, utils
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/test')

random.seed(1234)
np.random.seed(1234)

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
person_train_valid = np.load("person_train_valid.npy")
X_train_valid = np.load("X_train_valid.npy")
y_train_valid = np.load("y_train_valid.npy")
person_test = np.load("person_test.npy")

new_y_train_valid = y_train_valid - 769
new_y_test = y_test - 769

print('--------------------------------')
print('Preprocessing')
print('--------------------------------')


def data_prep(X, y, sub_sample, average, noise):
    total_X = None
    total_y = None

    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:, :, 0:500]
    print('Shape of X after trimming:', X.shape)

    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)

    total_X = X_max
    total_y = y
    print('Shape of X after maxpooling:', total_X.shape)

    # Averaging + noise
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average), axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)

    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    print('Shape of X after averaging+noise and concatenating:', total_X.shape)

    # Subsampling

    for i in range(sub_sample):
        X_subsample = X[:, :, i::sub_sample] + \
            (np.random.normal(0.0, 0.5,
                              X[:, :, i::sub_sample].shape) if noise else 0.0)

        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))

    print('Shape of X after subsampling and concatenating:', total_X.shape)
    return total_X, total_y


X_train_valid_prep, y_train_valid_prep = data_prep(
    X_train_valid, new_y_train_valid, 2, 2, True)

# Preprocessing the dataset

X_train_valid_prep, y_train_valid_prep = data_prep(
    X_train_valid, new_y_train_valid, 2, 2, True)
X_test_prep, y_test_prep = data_prep(X_test, new_y_test, 2, 2, True)

# print(X_train_valid_prep.shape)
# print(y_train_valid_prep.shape)
# print(X_test_prep.shape)
# print(y_test_prep.shape)


# Random splitting and reshaping the data

# First generating the training and validation indices using random splitting
ind_valid = np.random.choice(8460, 1500, replace=False)
ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))

# Creating the training and validation sets using the generated indices
(x_train,
 x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid]
(y_train,
 y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]
print('Shape of training set:', x_train.shape)
print('Shape of validation set:', x_valid.shape)
print('Shape of training labels:', y_train.shape)
print('Shape of validation labels:', y_valid.shape)


# Converting the labels to categorical variables for multiclass classification
# y_train = to_categorical(y_train, 4)
# y_valid = to_categorical(y_valid, 4)
# y_test = to_categorical(y_test_prep, 4)
# print('Shape of training labels after categorical conversion:',y_train.shape)
# print('Shape of validation labels after categorical conversion:',y_valid.shape)
# print('Shape of test labels after categorical conversion:',y_test.shape)

y_test = y_test_prep
# print(y_test.shape)


# Adding width of the segment to be 1
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_valid = x_valid.reshape(
    x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
x_test = X_test_prep.reshape(
    X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)
# print('Shape of training set after adding width info:', x_train.shape)
# print('Shape of validation set after adding width info:', x_valid.shape)
# print('Shape of test set after adding width info:', x_test.shape)


# Reshaping the training and validation dataset
# x_train = np.swapaxes(x_train, 1,3)
# x_train = np.swapaxes(x_train, 1,2)
# x_valid = np.swapaxes(x_valid, 1,3)
# x_valid = np.swapaxes(x_valid, 1,2)
# x_test = np.swapaxes(x_test, 1,3)
# x_test = np.swapaxes(x_test, 1,2)

print('Shape of training set after dimension reshaping:', x_train.shape)
print('Shape of validation set after dimension reshaping:', x_valid.shape)
print('Shape of test set after dimension reshaping:', x_test.shape)

x_total = np.concatenate((x_train, x_valid, x_test), axis=0)
y_total = np.concatenate((y_train, y_valid, y_test), axis=0)

new_x_total = copy.deepcopy(x_total)
new_y_total = copy.deepcopy(y_total)
new_y_total -= 2


def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


class CNNFeatureExtractor(MetaModule):

    def __init__(self, input_channels, num_classes, dropout_p):
        super(CNNFeatureExtractor, self).__init__()

        self.features = MetaSequential(
            MetaConv2d(input_channels, 25, (10, 1), padding='same'),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            MetaBatchNorm2d(25, False),
            nn.Dropout2d(dropout_p, inplace=False),
            MetaConv2d(25, 50, (10, 1), padding='same'),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            MetaBatchNorm2d(50, False),
            nn.Dropout2d(dropout_p, inplace=False),
            MetaConv2d(50, 100, (10, 1), padding='same'),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            MetaBatchNorm2d(100, False),
            nn.Dropout2d(dropout_p, inplace=False),
            MetaConv2d(100, 200, (10, 1), padding='same'),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            MetaBatchNorm2d(200, False),
            nn.Dropout2d(dropout_p, inplace=False)
        )

        self.classifier = MetaLinear(600, num_classes)

    def forward(self, inputs, params=None):
        features = self.features(
            inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(
            features, params=self.get_subdict(params, 'classifier'))
        return logits


class EEG(CombinationMetaDataset):
    def __init__(self, X, y, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None, dataset_transform=None):
        dataset = EEGClassDataset(X, y, meta_train=meta_train, meta_val=meta_val,
                                  meta_test=meta_test, meta_split=meta_split)
        super(EEG, self).__init__(dataset, num_classes_per_task,
                                  dataset_transform=dataset_transform)


class EEGClassDataset(ClassDataset):
    def __init__(self, X, y, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None,  transform=None):
        super(EEGClassDataset, self).__init__(meta_train=meta_train,
                                              meta_val=meta_val, meta_test=meta_test, meta_split=meta_split)

        if meta_train:
            self._labels = sorted(list(np.unique(y)))[:2]
        elif meta_test:
            self._labels = sorted(list(np.unique(y)))[2:]
        self._num_classes = len(self._labels)
        self._data = {}
        for i in range(self._num_classes):
            self._data[i] = X[y == i]

    def __getitem__(self, index):
        data = self._data[index % self._num_classes]

        return EEGDataset(index, data)

    @property
    def num_classes(self):
        return self._num_classes


class EEGDataset(Dataset):
    def __init__(self, index, data):
        super(EEGDataset, self).__init__(index)
        self.data = data
        self.activity = index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        eeg_data = self.data[index]
        target = self.activity
        return (eeg_data, target)


class EEGNormalDataset(TorchDataset):
    def __init__(self, X, y):
        super(EEGNormalDataset, self).__init__()
        self._X = X
        self._y = y

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, index):
        X = self._X[index, :]
        y = self._y[index]
        return torch.from_numpy(X).float(), torch.from_numpy(np.array([y])).long()


epochs = {8: 101, 16: 101}
epochs_other = {8: 101, 16: 101}
for shots in [8, 16]:
    print('--------------------------------')
    print(str(2*shots)+'-shot learning')
    print('--------------------------------')
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    dataset = EEG(x_total, y_total, num_classes_per_task=2, meta_train=True)
    dataset = ClassSplitter(dataset, shuffle=True,
                            num_train_per_class=shots, num_test_per_class=3*shots)
    dataset.seed(1234)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNFeatureExtractor(22, 2, 0.5)
    model.to(device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print('META-LEARNING')
    print('--------------------------------')
    # Training loop
    for itr in range(epochs[shots]):
        for batch_idx, batch in enumerate(dataloader):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device).float()
            train_targets = train_targets.to(device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device).float()
            test_targets = test_targets.to(device)

            outer_loss = torch.tensor(0., device=device)
            accuracy = torch.tensor(0., device=device)
            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(zip(train_inputs, train_targets,
                                                         test_inputs, test_targets)):
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=0.4,
                                                    first_order=True)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            outer_loss.backward()
            meta_optimizer.step()

            writer.add_scalar(str(shots)+"/meta/loss/train",
                              outer_loss.item(), itr)
            writer.add_scalar(str(shots)+"/meta/acc/train",
                              accuracy.item(), itr)

            if itr % 100 == 0:
                print('Iteration : ', itr+1, '\t', 'training_loss :',
                      outer_loss.item(), 'training_acc :', accuracy.item())

    lr = 0.01
    momentum = 0.9
    bs = 64
    p = 0.5
    tune_train_data = EEGNormalDataset(np.concatenate((new_x_total[new_y_total == 0][:shots], new_x_total[new_y_total == 1][:shots]), axis=0), np.concatenate(
        (new_y_total[new_y_total == 0][:shots], new_y_total[new_y_total == 1][:shots]), axis=0))
    tune_validation_data = EEGNormalDataset(np.concatenate((new_x_total[new_y_total == 0][-2000:-1000], new_x_total[new_y_total == 1][-2000:-1000]), axis=0), np.concatenate(
        (new_y_total[new_y_total == 0][-2000:-1000], new_y_total[new_y_total == 1][-2000:-1000]), axis=0))
    tune_test_data = EEGNormalDataset(np.concatenate((new_x_total[new_y_total == 0][-1000:], new_x_total[new_y_total == 1][-1000:]), axis=0), np.concatenate(
        (new_y_total[new_y_total == 0][-1000:], new_y_total[new_y_total == 1][-1000:]), axis=0))
    tune_train_data_loader = DataLoader(
        tune_train_data, batch_size=bs, shuffle=True, num_workers=0)
    tune_validation_data_loader = DataLoader(
        tune_validation_data, batch_size=bs, shuffle=False, num_workers=0)
    tune_test_data_loader = DataLoader(
        tune_test_data, batch_size=bs, shuffle=False, num_workers=0)

    model.eval()
    correct, total = 0, 0
    for X_te, y_te in tune_test_data_loader:
        X_te, y_te = X_te.to(device), y_te.to(device)
        output_test = model(X_te)
        total += X_te.shape[0]
        correct += (output_test.argmax(1) == y_te.squeeze(1)).sum()

    init_test_acc = correct.item()/total
    print('--------------------------------')
    print('Test accuracy wihtout finetuning: ', init_test_acc)
    print('--------------------------------')
    writer.add_scalar(str(shots)+"/tune/acc/itest", init_test_acc)

    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(1234)
    best_model = copy.deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(
        tune_train_data_loader), epochs=epochs_other[shots])
    best_acc = 0
    print('FINE-TUNING')
    print('--------------------------------')
    for i in range(epochs_other[shots]):
        model.train()
        for index, batch in enumerate(tune_train_data_loader):
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y.squeeze(1))
            loss.backward()
            optimizer.step()
        scheduler.step()
        writer.add_scalar(str(shots)+"/tune/loss/train",
                          loss.item(), i)
        model.eval()
        correct, total = 0, 0
        for X_val, y_val in tune_validation_data_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            output_val = model(X_val)
            total += X_val.shape[0]
            correct += (output_val.argmax(1) == y_val.squeeze(1)).sum()
        val_acc = correct.item()/total
        writer.add_scalar(str(shots)+"/tune/acc/val",
                          val_acc, i)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
        if i % 10 == 0 or i > epochs_other[shots]-10:
            print('Epoch : ', i+1, '\t', 'training_loss :',
                  loss.item(), 'validation_acc :', correct.item()/total)

    print('Best Validation Accuracy: '+str(best_acc))
    writer.add_scalar(str(shots)+"/tune/acc/bval", best_acc)
    print('--------------------------------')
    best_model.eval()
    correct, total = 0, 0
    for X_te, y_te in tune_test_data_loader:
        X_te, y_te = X_te.to(device), y_te.to(device)
        output_test = best_model(X_te)
        total += X_te.shape[0]
        correct += (output_test.argmax(1) == y_te.squeeze(1)).sum()
    print('Final Test Accuracy: '+str(correct.item()/total))
    writer.add_scalar(str(shots)+"/tune/acc/test", correct.item()/total)

writer.close()
