import torch
import torch.nn as nn

# with BlazePose
# class BiLSTMClassifier(nn.Module):
#     def __init__(self, input_size=78, hidden_size=73 , num_layers=2,
#                  dropout=0.2174, num_classes=4):
#         super().__init__()

#         self.bilstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=dropout if num_layers > 1 else 0.0,
#         )

#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

#     def forward(self, x):
#         # x: (batch, seq_len, input_size)
#         out, _ = self.bilstm(x)        # (batch, seq_len, hidden*2)
#         out = out[:, -1, :]            # take last timestep
#         out = self.dropout(out)
#         return self.fc(out)            # (batch, num_classes)


#with ZED
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=114, hidden_size=73, num_layers=2,
                 dropout=0.2174, num_classes=3):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.bilstm(x)        # (batch, seq_len, hidden*2)
        out = out[:, -1, :]            # take last timestep
        out = self.dropout(out)
        return self.fc(out)            # (batch, num_classes)
