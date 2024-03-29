import torch.nn as nn

class MyModel(nn.Module): # CNN
    def __init__(self, image_size, image_frame_num, action_num):
        super(MyModel, self).__init__()
        # 4*80*80
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = image_frame_num,
                      out_channels = 16,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),           
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        ) # 16*40*40
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16,
                      out_channels = 32,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        ) # 32*20*20
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 32,
                      out_channels = 64,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
        ) # 64*20*20
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 64,
                      out_channels = 128,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        ) # 128*10*10
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 128,
                      out_channels = 256,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        ) # 256*5*5
        self.out1 = nn.Linear(256*5*5, 64)
        self.out2 = nn.Linear(64, action_num)

              
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        x = self.out2(x)
        return x
        
    @staticmethod
    def initWeights(m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -0.01, 0.01)
            m.bias.data.fill_(0.01)    
        
# Model1:
   # 4*84*84
# self.conv1 = nn.Sequential(
#     nn.Conv2d(in_channels = image_frame_num,
#               out_channels = 16,
#               kernel_size = 5,
#               stride = 1,
#               padding = 2),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size = 2)
# ) # 16*42*42
# self.conv2 = nn.Sequential(
#     nn.Conv2d(in_channels = 16,
#               out_channels = 32,
#               kernel_size = 5,
#               stride = 1,
#               padding = 2),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size = 2)
# ) # 32*21*21
# self.conv3 = nn.Sequential(
#     nn.Conv2d(in_channels = 32,
#               out_channels = 64,
#               kernel_size = 3,
#               stride = 1,
#               padding = 1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size = 3)
# ) # 64*7*7
# self.out = nn.Linear(64*7*7, action_num)

# Model 2:
   # 4*84*84
# self.conv1 = nn.Sequential(
#     nn.Conv2d(in_channels = image_frame_num,
#                 out_channels = 16,
#                 kernel_size = 3,
#                 stride = 1,
#                 padding = 1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size = 2)
# ) # 16*42*42
# self.conv2 = nn.Sequential(
#     nn.Conv2d(in_channels = 16,
#                 out_channels = 32,
#                 kernel_size = 3,
#                 stride = 1,
#                 padding = 1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size = 2)
# ) # 32*21*21
# self.conv3 = nn.Sequential(
#     nn.Conv2d(in_channels = 32,
#                 out_channels = 64,
#                 kernel_size = 3,
#                 stride = 1,
#                 padding = 1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size = 3)
# ) # 64*7*7
# self.conv4 = nn.Sequential(
#     nn.Conv2d(in_channels = 64,
#                 out_channels = 128,
#                 kernel_size = 3,
#                 stride = 1,
#                 padding = 1),
#     nn.ReLU(),
# ) # 128*7*7
# self.conv5 = nn.Sequential(
#     nn.Conv2d(in_channels = 128,
#                 out_channels = 256,
#                 kernel_size = 3,
#                 stride = 1,
#                 padding = 1),
#     nn.ReLU(),
# ) # 256*7*7
# self.out1 = nn.Linear(256*7*7, 64)
# self.out2 = nn.Linear(64, 3)