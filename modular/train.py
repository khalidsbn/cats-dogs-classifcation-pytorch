"""
Trains a PyTorch image classification model.
"""
import os
import glob
import torch
import sys

from torchvision import transforms
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split

# Add the parent directory of 'modular' to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_setup import create_dataloaders
import engine, model_builder, utils

# Setup hyperparameters
NUM_EPOCHS = 1 # done
BATCH_SIZE = 100 # done
LEARNING_RATE = 0.001 # done

# Load train, valid and test data: Done
train_dir = "train"
test_dir = "test1"
train_list = glob.glob(os.path.join(train_dir, "*.jpg"))
test_list = glob.glob(os.path.join(test_dir, "*.jpg"))
train_list, valid_list = train_test_split(train_list, test_size=0.3, random_state=42)

# Setup device agnostic code: done
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
if device == "cuda":
  torch.cuda.manual_seed(42)

# Create transforms: done
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Create DataLoader's: done
train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
      train_list, valid_list, test_list, transform, BATCH_SIZE
)

# Create model: done
model = model_builder.CNN().to(device)

# Setup loss and optimizer: done
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# Start the timer: done
start_time = timer()

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=valid_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model to file
utils.save_model(model=model,
                 target_dir="models",
                 model_name="CNN_model.pth")
