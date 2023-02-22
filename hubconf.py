import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"


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

class NN(nn.Module):
  def __init__(self):
    super(NN,self).__init__()
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
  
def get_lossfn_and_optimizer(mymodel):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=1e-3)
    return loss_fn, optimizer

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

def train_d(dtldr,mdl,lfn,opt):
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
      
def test_d(dtldr,mdl,lfn):
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
  
def train(train_dataloader, test_dataloader, model1, loss_fn1, optimizer1, epochs=5):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train(train_dataloader, model1, loss_fn1, optimizer1)
        _test(test_dataloader, model1, loss_fn1)
    print("Done!")
    return model1

def save_model(model1,mypath="model.pth"):
    torch.save(model1.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

def load_model(mypath="model.pth"):
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    return model


def sample_test(model1, test_data):
    model1.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model1(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        
         

 #Completed!!
  
    
