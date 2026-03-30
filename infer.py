import torch
from model import MyMOLO
import config
from utils import plot_couple_examples, get_loaders
# 1. Initialize your specific model architecture first
model = MyMOLO(config_path=config.CONFIG_PATH).to(config.DEVICE)

# 2. Load the checkpoint file into memory
checkpoint = torch.load('checkpoint.pth.tar')

# 3. Extract the weights and load them into the model
model.load_state_dict(checkpoint['state_dict'])

# 4. Set the model to evaluation mode before using it for inference
model.eval()

# 5. Get the test loader and anchors
train_loader, test_loader, train_eval_loader = get_loaders(
    train_csv_path=config.TRAIN_DIR, test_csv_path=config.TEST_DIR
)

scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(config.DEVICE)

# 6. Plot couple examples
plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
