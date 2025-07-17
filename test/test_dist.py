import torch

# Define the list of values
values = torch.tensor([0, 1, 2, 3, 4])

# Define the lambda parameter for the exponential distribution
lambda_param = 0.3

# Generate weights using the exponential distribution
indices = torch.arange(3, dtype=torch.float32)
indices = 3 - indices - 1
weights = torch.exp(-lambda_param * indices)

# Normalize the weights to sum to 1
weights = weights / weights.sum()
print(weights)

for _ in range(10):
    # Sample from the list based on the exponential distribution weights
    sample_index = torch.multinomial(weights, num_samples=1, replacement=True)
    # sample_value = values[sample_index]

    print(f"Sampled Index: {sample_index+1}")
    # print(f"Sampled Value: {sample_value.item()}")
