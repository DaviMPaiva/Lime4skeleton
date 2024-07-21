import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torchvision.models.video import R3D_18_Weights


class UCF11Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.file_paths = []
        for label in self.classes:
            class_dir = os.path.join(root_dir, label)
            for video_folder in os.listdir(class_dir):
                if video_folder != 'Annotation':
                    class_folder_dir = os.path.join(class_dir, video_folder)
                    for video in os.listdir(class_folder_dir):
                        self.file_paths.append((os.path.join(class_folder_dir, video), label))
                
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        video_path, label = self.file_paths[idx]
        frames = self.load_video_frames(video_path)
        label = self.classes.index(label)
        if self.transform:
            frames = self.transform(frames)
        return frames, label
    
    def load_video_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames_array = np.array(frames)
        return torch.tensor(frames_array).permute(0, 3, 1, 2)  # T, C, H, W

# Set a seed for reproducibility
torch.manual_seed(42)

class ToFloatTensor:
    def __call__(self, img):
        return img.float()

class FrameTransform:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            ToFloatTensor(),  # Ensure tensor is float
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, video):
        transformed_frames = [self.transforms(frame) for frame in video]
        return torch.stack(transformed_frames).permute(1, 0, 2, 3)

def main():
    transform = FrameTransform()
    dataset = UCF11Dataset(root_dir='UCF11_updated_mpg', transform=transform)

    # Define the size of your training and testing sets
    train_size = int(0.8 * len(dataset))  # 80% training
    test_size = len(dataset) - train_size  # 20% testing

    # Split the dataset randomly
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create dataloaders for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    class ResNet3D(nn.Module):
        def __init__(self, num_classes=11):
            super(ResNet3D, self).__init__()
            self.resnet3d = models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)
            self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)
        
        def forward(self, x):
            return self.resnet3d(x)

    model = ResNet3D(num_classes=len(dataset.classes))
    #model = model.cuda()

    import torch.optim as optim
    import torch.nn.functional as F

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(dataloader):
                #inputs = inputs.cuda()
                #labels = labels.cuda()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

    train_model(model, train_dataloader, criterion, optimizer, num_epochs=25)

    def evaluate_model(model, dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                #inputs = inputs.cuda()
                #labels = labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')

    # Assuming you have a test dataloader
    evaluate_model(model, test_dataloader)

if __name__ == '__main__':
    main()