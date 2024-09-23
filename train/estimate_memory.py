
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_cold
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_cold

total_params = 13e9
largest_layer_params = 325e6
num_gpus_per_node = 8
num_nodes = 1

estimate_zero2_model_states_mem_needs_all_cold(
    total_params=total_params,
    num_gpus_per_node=num_gpus_per_node,
    num_nodes=num_nodes
)

estimate_zero3_model_states_mem_needs_all_cold(
    total_params=total_params,
    largest_layer_params=largest_layer_params,
    num_gpus_per_node=num_gpus_per_node,
    num_nodes=num_nodes
)
