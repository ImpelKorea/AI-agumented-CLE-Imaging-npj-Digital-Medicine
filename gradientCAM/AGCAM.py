import torch
from einops.layers.torch import Reduce, Rearrange
import torch.nn.functional as F
class AGCAM:
    """ Implementation of our method."""
    def __init__(self, model, attention_matrix_layer='before_softmax', attention_grad_layer='after_softmax',grad_mlp_layer="before_mlp" , head_fusion='sum', layer_fusion='sum', layer_num = 0, window_size=7):
        """
        Args:
            model (nn.Module): the Vision Transformer model to be explained
            attention_matrix_layer (str): the name of the layer to set a forward hook to get the self-attention matrices
            attention_grad_layer (str): the name of the layer to set a backward hook to get the gradients
            head_fusion (str): type of head-wise aggregation (default: 'sum')
            layer_fusion (str): type of layer-wise aggregation (default: 'sum')
        """
        self.model = model
        self.head = None
        self.width = None
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.attn_matrix = []
        self.grad_attn = []
        self.grad_mlp = []
        self.layer_num = layer_num
        self.window_size = window_size
        self.hook_handles = []
        # Grouping hooks by layer level
        self.layer_hooks = {}
        
        for layer_num, (name, module) in enumerate(self.model.named_modules()):
            print(name)
            if attention_matrix_layer in name or attention_grad_layer in name or grad_mlp_layer in name:
                # Extract the layer index
                print(f"Registering hooks for {name}")
                layer_level = int(name.split('.')[3])
                
                # Initialize list for specific layer if not already done
                if layer_level not in self.layer_hooks:
                    self.layer_hooks[layer_level] = {'attn': [], 'grad': []}
                
                if attention_matrix_layer in name:
                    # Register forward hook and append to layer's hooks
                    module.register_forward_hook(self.get_attn_matrix)
                    self.layer_hooks[layer_level]['attn'].append(name)
                    h = module.register_forward_hook(self.get_attn_matrix)
                    self.hook_handles.append(h)
                
                if attention_grad_layer in name:
                    # Register backward hook and append to layer's hooks
                    module.register_full_backward_hook(self.get_grad_attn)
                    self.layer_hooks[layer_level]['grad'].append(name)
                    h = module.register_full_backward_hook(self.get_grad_attn)
                    self.hook_handles.append(h)                   
                
                # if grad_mlp_layer in name:
                #     module.register_full_backward_hook(self.get_grad_mlp)
                #     self.layer_hooks[layer_level]['grad'].append(name)
                #     h = module.register_full_backward_hook(self.get_grad_mlp)
                #     self.hook_handles.append(h)
                
    def get_attn_matrix(self, module, input, output):
        # Collect attention matrices and print the shape for each layer
        # self.attn_matrix.append(output[:, :, 2:3, :]) # shape: [batch, num_heads, 1, num_patches]
        # self.attn_matrix.append(output)
        self.attn_matrix.append(output[:,:,self.layer_num:self.layer_num+1,:].detach())
        
        
        # out_min = output.min(dim=2, keepdim=True)[0]
        # out_max = output.max(dim=2, keepdim=True)[0]
        # normalized_output = (output - out_min) / (out_max - out_min + 1e-9)
        # self.attn_matrix.append(normalized_output.mean(dim=2).unsqueeze(2))
        
        
        
    def get_grad_attn(self, module, grad_input, grad_output):
        # Collect gradients and print the shape for each layer
        # self.grad_attn.append(grad_output[0][:, :, 2:3, :]) # shape: [batch, num_heads, 1, num_patches]
        # self.grad_attn.append(grad_output[0])
        self.grad_attn.append(grad_output[0][:,:,self.layer_num:self.layer_num+1,:].detach())
    
    def get_grad_mlp(self, module, grad_input, grad_output):
        self.grad_mlp.append(grad_output.detach())
        
        # out_min = grad_output[0].min(dim=2, keepdim=True)[0]
        # out_max = grad_output[0].max(dim=2, keepdim=True)[0]
        # normalized_output = (grad_output[0] - out_min) / (out_max - out_min + 1e-9)
        # normalized_output = torch.abs(grad_output[0])
        # self.grad_attn.append(normalized_output.mean(dim=2).unsqueeze(2))

        


    def generate(self, input_tensor, cls_idx=None):
        # Reset stored matrices and gradients
        
        def normalize_per_window(attn_map, window_size):
            batch_size, channels, _, window_h, window_w = attn_map.size()
            attn_map = attn_map.view(-1, window_h * window_w)
            min_vals = attn_map.min(dim=1, keepdim=True)[0]
            max_vals = attn_map.max(dim=1, keepdim=True)[0]
            attn_map = (attn_map - min_vals) / (max_vals - min_vals + 1e-5)
            attn_map = attn_map.view(batch_size, channels, 1, window_h, window_w)
            return attn_map
        
        def process_mask(mask, i):
            mask_shape = mask.shape
            
            mask = mask.permute(2, 0, 3, 1, 4).contiguous()  # (8, 8, 1, 7, 7) -> (1, 8, 7, 8, 7)
            mask = mask.view(-1, mask_shape[0] * self.window_size, mask_shape[1] * self.window_size)  # (-1, window * patch cnt, window * patch cnt)
            mask = mask.mean(dim=0).unsqueeze(0).unsqueeze(0)
            # roll
            if i % 2 == 0 and i not in [0, 1]:
                mask = torch.roll(mask, (3,3), dims=(1,2))
                
            mask = F.interpolate(mask, size=(224, 224), mode='bicubic', align_corners=False)
            # normalize 제거 (Global Normalize를 위해 Raw 값 유지)
            # mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
            mask = mask.squeeze(0)

            return mask

        self.attn_matrix = []
        self.grad_attn = []

        # Zero gradients and forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        _, prediction = torch.max(output, 1)
        confidence = F.softmax(output, dim=1)#.max(dim=1)[0]
        self.prediction = prediction  

        # Define loss based on class index or prediction
        loss = output[0, prediction[0]] if cls_idx is None else output[0, cls_idx]
        loss.backward()

        b, h, n, d = self.attn_matrix[0].shape
        self.head = h
        self.width = int((d-1)**0.5)

        # Combine attention matrices and gradients from layers
        self.attn_matrix.reverse()
        
        def reshape_attention_map(attn_map, window_size):
            attn_map = torch.abs(attn_map)
            # attn_map = torch.abs(attn_map)
            attn_map = attn_map.mean(dim=1) # multi head mean
            #
            attn_shape = attn_map.shape
            
            H, W = int(attn_shape[0]**0.5), int(attn_shape[0]**0.5)
            attn_map = attn_map.view(H, W, 1, window_size, window_size)
                
            return attn_map
        
        def reshape_grad_map(grad_map, window_size):
            # grad_map = torch.abs(grad_map)

            grad_map = grad_map.mean(dim=1)
            grad_shape = grad_map.shape  
            
            H, W = int(grad_shape[0]**0.5), int(grad_shape[0]**0.5)
            grad_map = grad_map.view(H, W, 1, window_size, window_size)
            
            return grad_map
        
        
        for i in range(len(self.attn_matrix)):
            self.attn_matrix[i] = reshape_attention_map(self.attn_matrix[i], self.window_size)
            self.grad_attn[i] = reshape_grad_map(self.grad_attn[i], self.window_size)
        

        mask_list = []
        attn_list = []
        grad_list = []
        
        
        for i in range(len(self.attn_matrix)):
            
            # attn = normalize_per_window(self.attn_matrix[i], 7)
            # gradient = normalize_per_window(self.grad_attn[i], 7)
            
            attn = self.attn_matrix[i]  
            gradient = self.grad_attn[i]
            
            
            # attn = torch.nn.functional.relu(attn)
            
            gradient = torch.nn.functional.relu(gradient)
            gradient = gradient / (gradient.max() + 1e-9)
            # gradient = torch.abs(gradient)
            # gradient = torch.sigmoid(gradient)
            
            # attn = torch.nn.functional.relu(attn)
            # attn = torch.abs(attn)
            
            attn = torch.sigmoid(attn)
            
            # softmax
            # attn = torch.nn.functional.softmax(attn, dim=-1)
            mask = gradient * attn
            
            
            attn_list.append(process_mask(attn, i))
            grad_list.append(process_mask(gradient, i))
            mask_list.append(process_mask(mask, i))
        
 
        # mask = torch.stack(mask_list, dim=0)
        # mask = mask.mean(dim=0)
        
        # print(f"Final mask shape: {mask.shape}")
        
        return prediction, mask_list, attn_list, grad_list, confidence

    # def __del__(self):
    #     # Remove hooks
    #     for layer_num, (name, module) in enumerate(self.model.named_modules()):
    #         if 'before_softmax' in name or 'after_softmax' in name:
    #             module._forward_hooks.clear()
    #             module._backward_hooks.clear()
    
    def clear_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []