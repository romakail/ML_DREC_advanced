
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self,
                 n_tokens,
                 n_cat_features,
                 num_features_1=1,
                 num_features_2=1,
                 num_features_3=1,
                 hid_size=64):
        super(ThreeInputsNet, self).__init__()
        
        self.concat_number_of_features = ((num_features_1 +
                                           num_features_2 +
                                           num_features_3) *
                                          hid_size)
        
        self.title_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.title_cnn = nn.Sequential()
        
#         self.title_cnn.add_module('reorder', Reorder())
        self.title_cnn.add_module('conv1',
                                  nn.Conv1d(
                                      in_channels=hid_size,
                                      out_channels=hid_size,
                                      kernel_size=2))
        self.title_cnn.add_module('relu1',
                                  nn.ReLU ())
        self.title_cnn.add_module('conv2',
                                  nn.Conv1d(
                                      in_channels=hid_size,
                                      out_channels=hid_size,
                                      kernel_size=2))
        self.title_cnn.add_module('relu2',
                                  nn.ReLU ())
        self.title_cnn.add_module('adapt_avg_pool',
                                  nn.AdaptiveAvgPool1d(
                                      output_size=num_features_1))
#         self.title_cnn.add_module('flatten',
#                                   Flatten())
         # <YOUR CODE HERE>        
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        
        self.full_cnn = nn.Sequential()
        
#         self.title_cnn.add_module('reorder', Reorder())
        self.full_cnn.add_module('conv1',
                                 nn.Conv1d(
                                     in_channels=hid_size,
                                     out_channels=hid_size,
                                     kernel_size=2))
        self.full_cnn.add_module('relu1',
                                 nn.ReLU ())
        self.full_cnn.add_module('conv2',
                                 nn.Conv1d(
                                     in_channels=hid_size,
                                     out_channels=hid_size,
                                     kernel_size=2))
        self.full_cnn.add_module('relu2',
                                 nn.ReLU ())
        self.full_cnn.add_module('adapt_avg_pool',
                                 nn.AdaptiveAvgPool1d(
                                     output_size=num_features_2))
#         self.full_cnn.add_module('flatten',
#                                  Flatten())
                                  
        # <YOUR CODE HERE>
        
        
        self.category_out = nn.Linear(in_features =n_cat_features,
                                      out_features= num_features_3*hid_size)# <YOUR CODE HERE>


        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features =self.concat_number_of_features,
                                     out_features=hid_size*2)
        self.last_ReLU   = nn.ReLU  ()
        self.final_dense = nn.Linear(in_features =hid_size*2,
                                     out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
                                 
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_cnn(title_beg)
        # <YOUR CODE HERE>
        
        full_beg  = self.full_emb (input2).permute((0, 2, 1))
        full  = self.full_cnn(full_beg)
        # <YOUR CODE HERE>        
        
        category = self.category_out(input3)
        # <YOUR CODE HERE>        
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.final_dense(self.last_ReLU(self.inter_dense(concatenated)))
        
        return out




class ThreeInputsNet_2(nn.Module):
    def __init__(self,
                 n_tokens,
                 n_cat_features,
                 num_features_3=1,
                 hid_size=64):
        super(ThreeInputsNet_2, self).__init__()
        
        self.concat_number_of_features = ((4 +
                                           4 +
                                           num_features_3) *
                                          hid_size)
        
#=======Title===========================================================================================

        self.title_emb   = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        
        self.title_cnn_1 = nn.Conv1d(
                               in_channels=hid_size,
                               out_channels=hid_size,
                               kernel_size=1)
        self.title_cnn_2 = nn.Conv1d(
                               in_channels=hid_size,
                               out_channels=hid_size,
                               kernel_size=3,
                               padding=1)
        self.title_cnn_3 = nn.Conv1d(
                               in_channels=hid_size,
                               out_channels=hid_size,
                               kernel_size=5,
                               padding=2)
        self.title_cnn_4 = nn.Conv1d(
                               in_channels=hid_size,
                               out_channels=hid_size,
                               kernel_size=7,
                               padding=3)
        
        self.title_relu  = nn.ReLU ()
        self.title_adapt = nn.AdaptiveAvgPool1d(
                               output_size=1)
        
#=======Full==================================================================================== 

        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        
        self.full_cnn_1 = nn.Conv1d(
                              in_channels=hid_size,
                              out_channels=hid_size,
                              kernel_size=1)
        self.full_cnn_2 = nn.Conv1d(
                              in_channels=hid_size,
                              out_channels=hid_size,
                              kernel_size=3,
                              padding=1)
        self.full_cnn_3 = nn.Conv1d(
                              in_channels=hid_size,
                              out_channels=hid_size,
                              kernel_size=5,
                              padding=2)
        self.full_cnn_4 = nn.Conv1d(
                              in_channels=hid_size,
                              out_channels=hid_size,
                              kernel_size=7,
                              padding=3)
        self.full_relu  = nn.ReLU ()
        self.full_adapt = nn.AdaptiveAvgPool1d(
                                       output_size=1)

#=======Categorical=====================================================================================
        
        self.category_out = nn.Linear(in_features =n_cat_features,
                                      out_features= num_features_3*hid_size)

#=====================================================================================================


        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features =self.concat_number_of_features,
                                     out_features=hid_size*2)
        self.last_ReLU   = nn.ReLU  ()
        self.final_dense = nn.Linear(in_features =hid_size*2,
                                     out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
                                 
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title_1 = self.title_cnn_1(title_beg)
        title_2 = self.title_cnn_2(title_beg)
        title_3 = self.title_cnn_3(title_beg)
        title_4 = self.title_cnn_4(title_beg)
        title = torch.cat([title_1,
                           title_2,
                           title_3,
                           title_4],
                          dim=1)
        title = self.title_relu(title)
        title = self.title_adapt(title)
       
    
    
        full_beg  = self.full_emb (input2).permute((0, 2, 1))
        full_1  = self.full_cnn_1 (full_beg)       
        full_2  = self.full_cnn_2 (full_beg)       
        full_3  = self.full_cnn_3 (full_beg)       
        full_4  = self.full_cnn_4 (full_beg)
        full  = torch.cat([full_1,
                           full_2,
                           full_3,
                           full_4],
                          dim=1)
        full = self.full_relu(full)
        full = self.full_adapt(full)
        
        category = self.category_out(input3)    
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.final_dense(self.last_ReLU(self.inter_dense(concatenated)))
        
        return out