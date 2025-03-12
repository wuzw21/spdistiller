import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import time
class ActivationModule:
    def __init__(self) :
        self.histograms_only: bool = False
        self.visualize_strategy: bool = False
        self.activations = None
        self.histograms = None
        self.file_path = None
        self.num_layers = 0
        self.num_weights = 0
    def set_visuable_strategy(self, s = True) :
        self.visualize_strategy = s
    def set_init_dict(self , num_layers , num_weights, file_path) :
        self.num_layers = num_layers
        self.num_weights = num_weights
        self.activations = [[[] for id in range(num_weights)] for _ in range(num_layers)]
        self.histograms= [[{} for id in range(num_weights)] for _ in range(num_layers)]
        self.file_path = file_path
        
    def fd_th(self, weight, sp) :
        tensor = weight.flatten().cpu().numpy()  # 将张量转换为 NumPy 数组
        threshold = np.percentile(tensor, q=sp*100)  # 计算分位数
        return threshold.item()        

    def fd_th_by_histogram(self, histogram, sp) :
        bin_centers = histogram["bin_centers"]
        counts = histogram["histogram"]

        # 取绝对值
        abs_bin_centers = torch.abs(bin_centers)

        # 重新分配直方图的计数
        unique_bins, inverse_indices = torch.unique(abs_bin_centers, return_inverse=True)
        new_counts = torch.zeros_like(unique_bins)
        new_counts.scatter_add_(0, inverse_indices, counts)

        # 重新计算累积分布
        cumulative_counts = torch.cumsum(new_counts, dim=0)
        total_count = cumulative_counts[-1]

        # 找到满足分位数的累积计数
        target_count = sp * total_count
        idx = torch.searchsorted(cumulative_counts, target_count)

        if idx == 0:
            threshold = unique_bins[0]
        elif idx == len(unique_bins):
            threshold = unique_bins[-1]
        else:
            lower_count = cumulative_counts[idx - 1]
            upper_count = cumulative_counts[idx]
            lower_value = unique_bins[idx - 1]
            upper_value = unique_bins[idx]

            fraction = (target_count - lower_count) / (upper_count - lower_count)
            threshold = lower_value + fraction * (upper_value - lower_value)

        return threshold.item()

    def cal_histograms(self, activations):
        
        flattened = activations.flatten().detach().to('cuda')
        
        k_low = int(0.01 * flattened.numel())
        k_high = int(0.99 * flattened.numel())
        lower_bound = torch.kthvalue(flattened, k_low).values
        upper_bound = torch.kthvalue(flattened, k_high).values

        mask = (flattened >= lower_bound) & (flattened <= upper_bound)
        filtered = flattened[mask]

        bins = 1000
        hist = torch.histc(filtered, bins=bins, min=lower_bound.item(), max=upper_bound.item())
        bin_edges = torch.linspace(lower_bound.item(), upper_bound.item(), steps=bins+1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return hist.cpu().numpy(), bin_edges.cpu().numpy(), bin_centers.cpu().numpy()

    def visualize_histogram(self, layer_idx, weight_idx):
        histograms_path = os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}", "histograms.pt")

        # 加载直方图数据
        histogram_data = torch.load(histograms_path, map_location='cpu', weights_only=True)
        histogram, bin_edges, bin_centers = histogram_data['histogram'], histogram_data['bin_edges'], histogram_data['bin_centers']

        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, histogram, width=(bin_edges[-1] - bin_edges[0]) / len(bin_edges), alpha=0.7, color='blue')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title(f"Histogram for Layer {layer_idx}, Weight {weight_idx}")

        # 先保存再显示
        plt.savefig(os.path.join(self.file_path, "histograms_png", f"layer_{layer_idx}-weight_{weight_idx}-histograms.png"))
        plt.show()
        plt.close() 

    
    def save_layer_weight(self, layer_idx, weight_idx) :
        if not self.activations[layer_idx][weight_idx]:
            print('None', self.activations[layer_idx][weight_idx])
            return 
        else :   
            layer_path = os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}")
            os.makedirs(layer_path, exist_ok=True)

            activations_path = os.path.join(layer_path, "activations.pt")
            combine_activations = torch.cat(self.activations[layer_idx][weight_idx], dim=0)
            if self.histograms_only == False :
                torch.save(combine_activations, activations_path)

            histograms_path = os.path.join(layer_path, "histograms.pt")
            histogram, bin_edges, bin_centers = self.cal_histograms(combine_activations)
            # print({'histogram': histogram, 'bin_edges': bin_edges, 'bin_centers': bin_centers})
            torch.save({'histogram': histogram, 'bin_edges': bin_edges, 'bin_centers': bin_centers}, histograms_path)

    def clear_layer_weight(self, layer_idx, weight_idx) :
        print('clear', layer_idx, weight_idx)
        self.activations[layer_idx][weight_idx] = []
        
    def grab_activations(self, x, layer_idx, weight_idx):
        if x.size(1) > 1:  # Check if seq_len > 1
            print('grab ', layer_idx, weight_idx, x.size())
            self.activations[layer_idx][weight_idx].append(x.detach().squeeze(0).cpu().float())
            
            self.save_layer_weight(layer_idx, weight_idx)
            self.clear_layer_weight(layer_idx, weight_idx)
            # 可视化直方图
            if self.visualize_strategy:
                start_time = time.time()
                self.visualize_histogram(layer_idx, weight_idx)
                visualize_time = time.time() - start_time  
                print(f"visualize_histogram ({layer_idx}, {weight_idx}) took {visualize_time:.6f} seconds")
    
    def save_activations(self):
        for layer_idx in range(self.num_layers):
            for weight_idx in range(self.num_weights) :
                self.save_layer_weight(layer_idx, weight_idx)
  
    def find_threshold(self, sp, output_path):
        thresholds = {}
        for layer_idx in range(self.num_layers):
            layer_thresholds = {}
            for weight_idx in range(self.num_weights):
                layer_path = os.path.join(self.file_path, f"layer_{layer_idx}", f"weight_{weight_idx}")
                if self.histograms_only == False :
                    activations_path = os.path.join(layer_path, "activations.pt")
                    weight = torch.load(activations_path)
                    threshold = self.fd_th(torch.abs(weight), sp)
                else :
                    histograms_path = os.path.join(layer_path, "histograms.pt")
                    histograms = torch.load(histograms_path)
                    threshold = self.fd_th_by_histogram(histograms, sp)
                
                layer_thresholds[weight_idx]=threshold
                print('find_threshold:  ', layer_idx, weight_idx, threshold)                
                
                del weight
                
                
            thresholds[layer_idx]=layer_thresholds

        # Save thresholds to output_path
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f"sparse-{sp}.json")
        with open(save_path,'w') as f:
            json.dump(thresholds, f)
        # print(thresholds)
        # torch.save(thresholds, )
        return thresholds