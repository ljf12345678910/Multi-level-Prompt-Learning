import torch.nn as nn
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, dropout_prob=0.4, k=2):
        super(Adapter, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(c_in, c_in // reduction, bias=False),
        #     # nn.BatchNorm1d(1370),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout_prob),
        #     nn.Linear(c_in // reduction, c_in, bias=False),
        #     # nn.BatchNorm1d(1370),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout_prob)
        # )
        self.k = k
        self.linear1 = nn.Linear(c_in, c_in // reduction, bias=False)
        self.linear2 = nn.Linear(c_in // reduction, c_in, bias=False)
        self.linear = nn.Linear(c_in, c_in, bias=False)
        self.drop = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        
    def forward(self, x):
        if self.k == 1:
            # x = x.unsqueeze(1)
            # x = self.conv(x)
            # x = x.squeeze(1)
            x = self.linear(x)
        if self.k == 2:
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.squeeze(1)
            x = self.linear1(x)
            x = self.relu(x)
            x = self.drop(x)
            x = self.linear2(x)
            x = self.relu(x)
            x = self.drop(x)
            
        return x
class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    def forward(self, x):
        x = self.linear(x)
        return x
class Adapters(nn.Module):
    def __init__(self, c_in, reduction=4, dropout_prob=0.4):
        super(Adapters, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            # nn.BatchNorm1d(c_in // reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(c_in // reduction, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(c_in // reduction, c_in, bias=False),
            # nn.BatchNorm1d(c_in),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        x = self.fc(x)
        return x