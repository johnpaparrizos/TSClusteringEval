### Still in progress ....


### 1. Common Parameters

| Hyperparameter      | Values to Try                         |
|---------------------|----------------------------------------|
| **latent_dim**      | `2, 8, 16, 32, 64`                     |
| **learning_rate**   | `1e-4, 5e-4, 1e-3, 5e-3`               |
| **batch_size**      | `32, 64, 128`                          |
| **pretrain_epochs** | `50, 100, 200`                         |
| **finetune_epochs** | `50, 100`                              |


### 2. Method-Specific Knobs

- **IDEC / DEC / DCN**
  - `update_interval`: `5, 10, 20`
  - `tol`: `1e-3, 1e-4`

- **DTC / DTCR**
  - convolutional kernel sizes: `(3,5), (5,7)`
  - number of TCN blocks: `2, 4`

- **SOM-VAE**
  - map size: `8×8, 16×16`
  - neighborhood radius σ: `0.5, 1.0, 1.5`

- **DEPICT**
  - dropout rate: `0.0, 0.2, 0.5`
  - auxiliary loss weight: `0.1, 0.5`

- **SDCN**
  - GCN layers: `1, 2, 3`
  - hidden units: `32, 64, 128`

- **ClusterGAN**
  - noise_dim: `10, 20`
  - discriminator_steps per generator_step: `1, 2`

- **VADE**
  - KL-loss weight: `0.1, 0.5, 1.0`

- **Foundation Models (MOMENT, OFA, CHRONOS)**
  - frozen layers: all but last, last 2, last 4
