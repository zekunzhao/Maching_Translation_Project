import torch
torch.backends.cudnn.enabled = False
print(torch.backends.cudnn.is_available())
rnn = torch.nn.LSTM(10,10)  # same error with e.g. torch.nn.GRU(10,10,1)
rnn.cuda()
