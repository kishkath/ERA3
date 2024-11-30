import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()



        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1, bias=False),

            nn.BatchNorm2d(16),

            nn.ReLU())

        """

        out_features = (28+2-3)/1 + 1 = 28

        Jin = 1, S=1, Jout = 1

        Rfout = Rin + (K-1)*Jin = 1 + (3-1)*1 = 3

        """



        self.conv2 = nn.Sequential(

            nn.Conv2d(16, 32, 3, padding=0, bias=False),

            nn.BatchNorm2d(32),

            nn.ReLU())

        """

        out_features = (28-3)/1 + 1 = 26

        Jin = 1, S = 1, Jout = 1

        Rfout = Rin + (K-1)*Jin = 3 + (3-1)*1 = 5

        """

        self.maxPool = nn.MaxPool2d((2, 2))

        """

        out_features = (26-2)/2 + 1 = 13

        Jin = 1, S = 2, Jout = 2

        Rfout = Rin + (K-1)*Jin = 5 + (3-1)*1 = 7

        """

        self.conv3 = nn.Sequential(

            nn.Conv2d(32, 32, 3, padding=0, bias=False),

            nn.BatchNorm2d(32),

            nn.ReLU())

        """

        out_features = (13-3+2) + 1 = 11

        Jin = 2, S = 1, Jout = 2

        Rfout = Rin + (K-1)*Jin = 7 + (3-1)*2 = 11

        """

        self.conv4 = nn.Sequential(

            nn.Conv2d(32, 20, 3, padding=1, bias=False),

            nn.BatchNorm2d(20),

            nn.ReLU())

        """

        out_features = (11-3+2) + 1 = 11

        Jin = 2, S = 1, Jout = 2

        Rfout = Rin + (K-1)*Jin = 11 + (3-1)*2 = 15

        """

        self.maxPool1 = nn.MaxPool2d((2, 2))

        """

        out_features = (11-2)/2 + 1 = 5

        Jin = 2, S = 2 ,Jout = 4

        Rfout = Rin + (K-1)*Jin = 15 + (2-1)*2 = 17

        out_features = ()"""



        self.conv5 = nn.Sequential(

            nn.Conv2d(20, 16, 1, padding=0, bias=False),

            nn.BatchNorm2d(16),

            nn.ReLU())

        """

        out_features = (5-3)+1 = 3

        Jin = 4, S = 1, Jout = 4

        Rfout = Rin + (K-1)*Jin = 17 + (3-1)*4 = 25

        """

        self.conv6 = nn.Sequential(nn.Conv2d(16, 10, 3),

                                   nn.BatchNorm2d(10),

                                   nn.ReLU())



        self.gap = nn.AvgPool2d(3)



        """

        out_features = (3-3) + 1 = 1

        Jin = 4, S = 1 , Jout = 4



        Rfout = Rin +(K-1)*Jin = 33

        """



        self.dropout = nn.Dropout2d(0.01)



    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.maxPool(x)

        # x = self.dropout(x)

        x = self.conv3(x)

        x = self.dropout(x)

        x = self.conv4(x)

        x = self.maxPool1(x)

        x = self.conv5(x)

        # x = self.dropout(x)

        x = self.conv6(x)

        x = self.gap(x)



        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)




def get_summary(model, input_size=(1, 28, 28)):
    """Calculate total number of parameters in the model"""
    from torchsummary import summary
    summary(model, input_size=input_size)
    return sum(p.numel() for p in model.parameters()) 