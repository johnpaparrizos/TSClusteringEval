
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
 

### 3. Encoder–Decoder Architecture

- **IDEC / DEC / DCN**
  - **encoder_layers**: `1, 2`
  - **hidden_units (per layer)**: `64, 128`
  - **activation**: `ReLU, LeakyReLU`
  - **decoder_layers**: mirror encoder_layers

- **DTC / DTCR**
  - **TCN blocks**: `2, 3`
  - **hidden_units**: `64, 128`
  - **kernel_sizes**: `(3,5), (5,7)`
  - **dropout_rate**: `0.0, 0.2`

- **SOM-VAE**
  - **encoder_layers**: `1, 2`
  - **hidden_units**: `128, 256`
  - **rnn_type**: `LSTM, GRU`
  - **decoder_layers**: mirror encoder

- **DEPICT**
  - **conv_layers (per block)**: `2, 3`
  - **filters per conv**: `32, 64`
  - **activation**: `ReLU`
  - **decoder symmetric**: yes

- **SDCN**
  - **GCN layers**: `1, 2`
  - **hidden_units**: `64, 128`
  - **activation**: `ReLU, LeakyReLU`
  - **decoder_layers**: `1, 2`

- **ClusterGAN**
  - **generator_layers**: `2, 3`
  - **latent_dim**: matches grid
  - **hidden_units**: `128, 256`
  - **activation**: `ReLU, LeakyReLU`

- **VADE**
  - **encoder_layers**: `1, 2`
  - **hidden_units**: `64, 128`
  - **rnn_type**: `LSTM, GRU`
  - **decoder_layers**: mirror encoder

- **Foundation Models (MOMENT, OFA, CHRONOS)**
  - **frozen backbone**: last `1, 2` layers trainable
  - **head hidden_units**: `128, 256`
  - **head layers**: `1, 2`
