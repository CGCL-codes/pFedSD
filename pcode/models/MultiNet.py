import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = ['StuNet', 'TransferNet','MutualNet']
__all__ = ['TransferNet']

# class MutualNet(nn.Module):
#     def __init__(self, models, input_channel=64, factor=8):
#         super().__init__()
#         self.num_models = len(models)
        
#         for i, model in enumerate(models):
#             setattr(self, 'model'+str(i), model)
        
#         # self.query_linear = nn.Linear(input_channel, input_channel//factor, bias = False)
#         # self.key_linear = nn.Linear(input_channel, input_channel//factor, bias = False)
        

#     def forward(self, x):
#         # no attention way
#         # out = self.model0(x) 
#         # out = out.unsqueeze(-1)
#         # for i in range(1, self.num_models):
#         #     temp_out = getattr(self, 'model'+str(i))(x)
#         #     temp_out = temp_out.unsqueeze(-1)
#         #     out = torch.cat([out, temp_out],-1)        
#         # return out    
                           
#         pro = self.model0(x)                        # B X num_classes
#         # x_f = self.model0.activations[3]            # B X 64
#         # x_f = F.normalize(x_f)
#         # proj_query = self.query_linear(x_f)             # B X input_channel//factor
#         # proj_key = self.key_linear(x_f)                 # B X input_channel//factor

#         # proj_query = proj_query[:, None, :]           # B X 1 X input_channel//factor
#         # proj_key = proj_key[:, None, :]
#         query = pro[:, None, :]
#         key = pro[:, None, :]
#         pro = pro.unsqueeze(-1)                      # B X num_classes X 1 

#         for i in range(1, self.num_models):
#             temp_pro = getattr(self, 'model'+str(i))(x)
#             temp_x_f = getattr(self, 'model'+str(i)).activations[-1]
#             temp_x_f = F.normalize(temp_x_f)
#             # x_f_q = self.query_linear(temp_x_f)                # B x input_channel//factor
#             # x_f_k = self.key_linear(temp_x_f)                  # B x input_channel//factor
#             # proj_query = torch.cat([proj_query, x_f_q[:,None,:]], 1) # B X num_models-1 X input_channel//factor
#             query = torch.cat([query, temp_x_f[:,None,:]], 1) 
#             # proj_key = torch.cat([proj_key, x_f_k[:,None,:]], 1)
#             key = torch.cat([key, temp_x_f[:,None,:]], 1) 
#             temp_pro = temp_pro.unsqueeze(-1)          
#             pro = torch.cat([pro, temp_pro],-1)    # B X num_classes X num_models-1        
#         energy = torch.bmm(query, key.permute(0,2,1)) # B X num_models-1 X num_models-1
#         attention = F.softmax(energy, dim = -1) 
#         x_m = torch.bmm(pro, attention.permute(0,2,1)) # B X num_classes X num_models-1
        
#         return pro, x_m
    
#     def getWeightMatrix(self):
#         return self.query_linear.state_dict(), self.key_linear.state_dict()

    
class TransferNet(nn.Module):
    def __init__(self, models, query_weight=None, key_weight=None, weight_vector=None, input_channel=64, factor=8):
        super().__init__()
        self.num_models = len(models)
        
        for i, model in enumerate(models):
            setattr(self, 'model'+str(i), model)
        
        # self.query_linear = nn.Linear(input_channel, input_channel//factor, bias = False)
        # self.key_linear = nn.Linear(input_channel, input_channel//factor, bias = False) 
        # if query_weight is not None and key_weight is not None:
        #     self.query_linear.load_state_dict(query_weight)
        #     self.key_linear.load_state_dict(key_weight)
        
        # self.weight_vector=F.softmax(weight_vector) # 1 X num_models-1
        self.attn = torch.zeros(self.num_models).cuda()
        self.save_attn = False
        self.attn_matrix = torch.zeros(self.num_models).unsqueeze(0).cuda()
        # self.scale = 50.0

    def forward(self, x):                       
        pro = self.model0(x)                        # B X num_classes
        # x_f = self.model0.activations[-1]            # B X 64
        # x_f = F.normalize(x_f)
        # proj_key = self.key_linear(x_f)                 # B X input_channel//factor

        # proj_key = proj_key[:, None, :]
        key = pro[:, None, :]
        pro = pro.unsqueeze(-1)                      # B X num_classes X 1 

        for i in range(1, self.num_models-1):
            temp_pro = getattr(self, 'model'+str(i))(x)
            # temp_x_f = getattr(self, 'model'+str(i)).activations[-1]
            # temp_x_f = F.normalize(temp_x_f)
            # x_f_k = self.key_linear(temp_x_f)                  # B x input_channel//factor
            # proj_key = torch.cat([proj_key, x_f_k[:,None,:]], 1) # B X num_models-1 X input_channel//factor
            key = torch.cat([key, temp_pro[:,None,:]], 1)
            temp_pro = temp_pro.unsqueeze(-1)
            pro = torch.cat([pro, temp_pro],-1)    # B X num_classes X num_models-1      
        
        temp_pro = getattr(self, 'model'+str(self.num_models-1))(x) # B X num_classes
        # temp_x_f = getattr(self, 'model'+str(self.num_models-1)).activations[-1] # B X 64
        # temp_x_f = F.normalize(temp_x_f)
        
        with torch.no_grad():
            # query_stu = self.query_linear(temp_x_f)[:, None, :]                 # B X 1 X input_channel//factor
            query_stu = temp_pro[:, None, :]                 # B X 1 X input_channel//factor
            key = torch.cat([key, temp_pro[:,None,:]], 1)
            pro = torch.cat([pro, temp_pro.unsqueeze(-1)],-1)
            energy_stu = torch.bmm(query_stu, key.permute(0,2,1))         # B X 1 X num_models-1
            # attn_stu = F.softmax(energy_stu / self.scale, dim = -1)
            energy_stu_pos = F.relu(energy_stu)
            attn_stu = energy_stu_pos / torch.sum(energy_stu_pos, dim=-1, keepdim=True)
        
            if self.save_attn:
                # print(attn_stu)
                avg = torch.mean(energy_stu, dim = 0).squeeze()
                # print(avg)
                self.attn += avg #num_models
                self.attn_matrix = torch.cat([self.attn_matrix, avg[None,:]],0)

            # attn_stu = self.weight_vector.repeat(pro.size(0),1,1) 
            attn_target_stu = torch.bmm(pro, attn_stu.permute(0,2,1)).squeeze(-1)  # B X num_classes  
            avg_logit = 0
            for i in range(self.num_models):
                avg_logit += pro[:,:,i]

        # return temp_pro, avg_logit/self.num_models
        return temp_pro, attn_target_stu

# class StuNet(nn.Module):
#     def __init__(self, models, input_channel=64, factor=8, attn_target_stu_bool = True):
#         super(StuNet, self).__init__()
        
#         self.num_models = len(models)
#         for i, model in enumerate(models):
#             model.save_activations = True
#             setattr(self, 'model'+str(i), model)

#         self.query_linear = nn.Linear(input_channel, input_channel//factor, bias = False)
#         self.key_linear = nn.Linear(input_channel, input_channel//factor, bias = False)

#         self.attn_target_stu_bool = attn_target_stu_bool
#         self.attn_target_stu = None
#         self.attn_stu_to_all = None
#         self.attn = torch.zeros(self.num_models).cuda()
#         # self.attn = None
            
#     def forward(self, x):
#         # x_f, pro = self.stu0(x)                         
#         pro = self.model0(x)                        # B X num_classes
#         x_f = self.model0.activations[-1]            # B X 64
#         proj_query = self.query_linear(x_f)             # B X input_channel//factor
#         proj_key = self.key_linear(x_f)                 # B X input_channel//factor

#         proj_query = proj_query[:, None, :]           # B X 1 X input_channel//factor
#         proj_key = proj_key[:, None, :]
#         pro = pro.unsqueeze(-1)                      # B X num_classes X 1 

#         for i in range(1, self.num_models-1):
#             # temp_x_f, temp_pro = getattr(self, 'stu'+str(i))(x)
#             temp_pro = getattr(self, 'model'+str(i))(x)
#             temp_x_f = getattr(self, 'model'+str(i)).activations[-1]
#             x_f_q = self.query_linear(temp_x_f)                # B x input_channel//factor
#             x_f_k = self.key_linear(temp_x_f)                  # B x input_channel//factor
#             temp_pro = temp_pro.unsqueeze(-1)
#             # B X num_models-1 X input_channel//factor
#             proj_query = torch.cat([proj_query, x_f_q[:,None,:]], 1) 
#             proj_key = torch.cat([proj_key, x_f_k[:,None,:]], 1)
#             # B X num_classes X num_models-1
#             pro = torch.cat([pro, temp_pro],-1)          
#         energy = torch.bmm(proj_query, proj_key.permute(0,2,1)) # B X num_models-1 X num_models-1
#         attention = F.softmax(energy, dim = -1) 
#         x_m = torch.bmm(pro, attention.permute(0,2,1)) # B X num_classes X num_models-1
        
#         temp_pro = getattr(self, 'model'+str(self.num_models-1))(x) # B X num_classes
#         if self.attn_target_stu_bool:
#             temp_x_f = getattr(self, 'model'+str(self.num_models-1)).activations[-1] # B X 64
#             query_stu = self.query_linear(temp_x_f)[:, None, :]                 # B X 1 X input_channel//factor
#             key_stu = self.key_linear(temp_x_f)[:, None, :]
#             energy_stu = torch.bmm(query_stu, proj_key.permute(0,2,1))         # B X 1 X num_models-1
#             attn_stu = F.softmax(energy_stu, dim = -1)
#             self.attn_target_stu = torch.bmm(pro, attn_stu.permute(0,2,1)).squeeze(-1)  # B X num_classes   

#         # if self.attn is None: 
#             proj_key_all = torch.cat([proj_key, key_stu], 1)        # B X num_models X input_channel//factor
#             energy_stu_to_all = torch.bmm(query_stu, proj_key_all.permute(0,2,1))         # B X 1 X num_models
#             attn_stu_to_all = F.softmax(energy_stu_to_all.squeeze(1), dim = -1)
#             # self.attn_stu_to_sall = F.tanh(energy_stu_to_all.squeeze(1))

#             # self.attn = torch.mean(attn_stu_to_all, dim = 0) #num_models
#             self.attn += torch.mean(attn_stu_to_all, dim = 0) #num_models
        
#         return pro, x_m, temp_pro
    
#     # def updateModels(self, models):
#     #     self.num_models = len(models)
#     #     self.attn = torch.zeros(self.num_models).cuda()
#     #     # self.attn = None
#     #     for i, model in enumerate(models):
#     #         model.save_activations = True
#     #         setattr(self, 'model'+str(i), model)
        
#     def getModelCount(self):
#         return self.num_models

#     def getAttnTargetStu(self):
#         return self.attn_target_stu
    
#     def getAttn(self):
#         return self.attn

# if __name__ == "__main__":
    # models = {0:"net1", 1:"net2", 2:"net3"}
    # multinet = StuNet(models)
    # print(multinet.getLastModel())
    # a = torch.randn(4, 4)
    # print(a)
    # b = torch.mean(a, dim=1)
    # print(b[0])
    # print(b.shape)
    # import numpy as np
    # t = torch.ones(5, 5)
    # m = np.array([1, 3])
    # s = torch.Tensor([20,20])
    # t[1][m-1] += s
    # print(t)
    # t[1,:]=3
    # print(t.size())
    # data = torch.Tensor(torch.Size([30]))
    # print(data.tolist())
    # import numpy as np
    # a = np.array([1,12,3,2,24])
    # s = a.sum()
    # a = a/s
    # a.sort()
    # print(a)
    # aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
    # x = np.random.choice(aa_milne_arr, 3, p=[0.5, 0.1, 0.1, 0.3])
    # print(x)
    # a = torch.ones(6, 5, 4)
    # print(a)
    # b = torch.rand(1, 4)
    # print(b)
    # b = b.repeat(6, 1, 1)
    # print(b.size(0))
    # print(b)

    # c = torch.bmm(a, b.permute(0,2,1))
    # print(c)
    # print(c.size())

