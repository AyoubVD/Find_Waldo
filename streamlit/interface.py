import torch

if __name__ == "__main__":
    # Load model  
    PATH = "ptModel/model.pt"
    device = torch.device('cpu')
    model = torch.load(PATH, map_location=device)
    state_dict = model.FeedForwardNet()