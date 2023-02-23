import numpy as np

architectures = [[2000, 1000, 500, 500, 250, 250, 50],
                 [2000, 1000, 250, 50],
                 [2000, 1000, 50],
                 [64, 64, 32, 32, 16, 16, 10],
                 [64, 64, 32, 10],
                 [64, 64, 10],
                 [256, 256, 128, 128, 32, 20],
                 [256, 256, 128, 20],
                 [256, 128, 20]]

for j, arch in enumerate(architectures):
    params = []
    for i in range(128):
        params.append([arch])

    params = np.array(params)
    np.save('arch' + str(j+1) + '.npy', params)
