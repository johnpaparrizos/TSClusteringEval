import numpy as np


num_layers = [3, 4, 5]
config = 1
for j, arch in enumerate(num_layers):
    for lambda_1 in [3, 5, 7]:
        params = []
        for i in range(128):
            params.append([50, lambda_1, arch])

        params = np.array(params, dtype=object)
        np.save('tcnae_arch' + str(config) + '.npy', params)
        config = config + 1
