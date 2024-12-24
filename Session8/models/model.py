## Create a new model 
# C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10

# n_out formula = (n_in + 2P - K)/S + 1
# Rf_out = Rin + (K - 1)  * Jin
# with dilation n_out = (n_in + 2P - d(K-1)-1)/S + 1

# CIFAR Image resolution: 32 with 3x3(RGB Channels)

dropout_val = 0.1

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, dilation=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        ) 
        # Layer1: nout = (32 + 2 - 3)/1 + 1 = 32, Rf_out = 1 + (3-1)*1 = 3
        # Layer2: nout = (32 + 0 - 2(3-1)-1) + 1 = 30, RF_out = 3 + 2*(3-1) = 7


        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        # Layer3: nout = (30 + 2 - 3) + 1 = 30, Rf_out = 7 + (3-1)*1 = 9
        # Layer4: nout = (30 + 2 - 3)/2 + 1 = 15, Jin = 1, Rf_out = 9 + (3-1)*1 = 11

        self.conv_block3 = nn.Sequential(
            
            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3, stride=1, groups=128),
            # point wise
            nn.Conv2d(in_channels=128,out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        # Layer5: nout = (15 + 2 - 3) + 1 = 15, Rf_out = 11 + (3-1)*2 = 15
        # Layer6: nout = (15-3) +  1 = 13, Rf_out = 15 + (3-1)*2 = 19
        # Layer7: nout = 13, Rf_out = 19
        # Layer8: nout = (13 - 3)/2 + 1 = 6, Rf_out = 19 + (3-1)*2 = 23, Jout = 2*2 = 4s

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Dropout(dropout_val),
        )

        # Layer9: (6 + 2 - 3) + 1 = 6, Rf_out = 23 + (3-1)*4 = 31
        # Layer10: (6 + 2 - 3) + 1 = 6, Rf_out = 31  + (3-1)*4 = 39
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Layer11: (6 - 6) + 1 = 1, RF_out = 39 + (7-1)*4 = 63
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.gap(x)
        x = x.view(-1,32)
        x = self.fc(x)
        return x
        # return F.log_softmax(x,dim=-1)

