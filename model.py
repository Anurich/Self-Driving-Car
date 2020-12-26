from dataset import *
import torch
from sklearn.metrics import accuracy_score
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.model  = torchvision.models.vgg16(pretrained=True)
        self.model  =  nn.Sequential(*list(self.model.children())[:-2])
        for param in self.model.parameters():
            param.requires_grad = False
        # now we can use dense layers
        self.dense1 =  nn.Linear(25088,100)
        self.dense2 =  nn.Linear(100,50)
        self.dense3 =  nn.Linear(50,10)
        self.output =  nn.Linear(10,1)

    def forward(self,x):
        features = self.model(x)
        output   = F.elu(self.dense1(features.view(x.size(0),-1)))
        output   = F.elu(self.dense2(output))
        output   = F.elu(self.dense3(output))
        output   = self.output(output)
        return output




def read_data():
    transformation =  torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.2,0.75),
        torchvision.transforms.ToTensor(),
    ])

    data =dataset(transformation)
    train_size = 0.7
    val_size   = 0.2
    test_size  = 0.1

    indices = GroupShuffleSplit(n_splits=1,train_size=0.8,random_state=42)
    indices_train_validation = indices.split(data.imageName,data.newSteeringAngle,groups=data.newSteeringAngle)

    train,validation =  list(indices_train_validation)[0]

    indices_test = GroupShuffleSplit(n_splits=1,test_size=0.1,train_size=0.9,random_state=42)

    new_img = list(map(lambda x: data.imageName[x], train))
    new_steer = list(map(lambda x: data.newSteeringAngle[x],train))
    indices_train_test = indices_test.split(new_img,new_steer,groups=new_steer)
    train,test  = list(indices_train_test)[0]

    train_indices = torch.utils.data.SubsetRandomSampler(train)
    val_indices = torch.utils.data.SubsetRandomSampler(validation)
    test_indices = torch.utils.data.SubsetRandomSampler(test)

    ''' indices = np.arange(0,len(data.steeringAngle)+1)'''
    '''train_indices, test_indices, val_indices = indices[:40210],indices[40211:44677],indices[44677:]'''

    batch_sample_train = torch.utils.data.BatchSampler(train_indices,batch_size=32,drop_last=False)
    batch_sample_val  = torch.utils.data.BatchSampler(val_indices,batch_size=32,drop_last=False)
    batch_sample_test = torch.utils.data.BatchSampler(test_indices,batch_size=32,drop_last=False)
    train_data = torch.utils.data.DataLoader(data,batch_sampler=batch_sample_train)
    val_data  = torch.utils.data.DataLoader(data,batch_sampler=batch_sample_val)
    test_data = torch.utils.data.DataLoader(data,batch_sampler=batch_sample_test)

    return train_data, val_data, test_data

def main():
    net = model()
    criterian = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    train_data_loader, val_data_loader, test_data_loader =  read_data()
    iteration = 1
    prev_val_loss= 0.0
    prev_train_loss =0.0
    for i in tqdm(range(iteration)):
        total_loss = 0.0
        for j,data in enumerate(train_data_loader):
            imgX,steerAng = data
            steerAng = steerAng.type(torch.FloatTensor)
            net.zero_grad()
            prediction = net(imgX).squeeze()
            loss       = criterian(prediction,steerAng)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if j % 20 == 0 and j != 0:
                if not os.path.isdir("model_weights"):
                    os.mkdir("model_weights")
                torch.save({
                    "epoch":i,
                    "model_state_dict":net.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                },"model_weights/weight_epoch_"+str(i))


                # evalaution for val data
                net.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for k,data_val in enumerate(val_data_loader):
                        valX,valY = data_val
                        valY = valY.type(torch.FloatTensor)
                        prediction_val = net(valX).squeeze()
                        val_loss  = criterian(prediction_val,valY)
                        total_val_loss += val_loss.item()
                        if k%20 == 0 and k!=0:
                            break
                prev_val_loss = total_val_loss/20
                prev_train_loss = total_loss/20
                print("The training loss after epoch {} and step {} is {}, And validation loss {}".format(str(i),str(j),str(total_loss/20),str(total_val_loss/20)))
                total_loss = 0.0
if __name__ == "__main__":
    weight_folder = "model_weights/"
    if not os.path.isdir(weight_folder):
        main()
    else:

        with torch.no_grad():
            _,_,test= read_data()
            net =model()
            checkpoint = torch.load(weight_folder+"weight_epoch_0")
            net.load_state_dict(checkpoint["model_state_dict"])
            net.eval()

            x,y = next(iter(test))
            prediction = net(x)

            plt.plot(prediction.numpy(),label="predicted_SteerAngle")
            plt.plot(y,label="Actual Steer Angle")
            plt.legend(frameon=False, loc='lower center', ncol=2)
            plt.show()

