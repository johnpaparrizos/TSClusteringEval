import numpy as np


dilations = [[1], [8]]
config = 1
for j, arch in enumerate(dilations):
    for lambda_1 in [1, 1e-2]:
        params = []
        for i in range(128):
            params.append([[30], [lambda_1], arch])

        params = np.array(params, dtype=object)
        np.save('dtcr_arch' + str(config) + '.npy', params)
        config = config + 1
