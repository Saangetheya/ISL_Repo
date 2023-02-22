import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SDG(model.parameters(),lr = 1e-3)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

class NN(nn.module):
  def __init__(self):
    super(NeuralNetwork,self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28,512),
      nn.ReLU(),
      nn.Linear(512,512),
      nn.ReLU(),
      nn.Linear(512,10)
    )
  def forward(self,x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
  

def load_data():
  train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
  )
  test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
  )
  return train_data,test_data


def create_DL(trd,tsd,btch=64):
  trdl = DataLoader(trd,batch_size=btch)
  tsdl = DataLoader(tsd,batch_size=btch)
  
  return trdl,tsdl


def get_model():
  return NN().to(device)

def train_d(dtldr,mdl,lfn=loss_fn,opt=optimizer):
  size = len(dtldr.dataset)
  mdl.train()
  for batch, (X,y) in enumerate(dtldr):
    X,y = X.to(device),y.to(device)
    pred = mdl(X)
    loss = lfn(pred,y)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if batch %100==0:
      loss,current = loss.item(),batch*len(X)
      print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
      
def test_d(dtldr,mdl,lfn=loss_fn):
  size = len(dtldr.dataset)
  num_b = len(dtldr)
  mdl.eval()
  test_loss,correct = 0,0
  with torch.no_grad():
    for X,y in dtldr:
      X,y = X.to(device),y.to(device)
      pred = model(X)
      test_loss += lfn(pred,y).item()
      correct += (pred.argmax(1)==y).type(torch.float).sum().item()
  test_loss /= num_b
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  
 def train(train_dataloader, test_dataloader, epochs=5):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

def save_model(mypath="model.pth"):
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

def load_model(mypath="model.pth"):
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))


def sample_test(model, test_data):
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
         
  
  
    
