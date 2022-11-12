## Experiment Results

### Modification
Adding a self-attention to the network before the ffns to escalate the inner relations between different datapoints from different rays.
The experiment has been done with fern dataset

### Testing Results
Performance comparison between original model and model_w_attention

|                 | avg psnr | avg rendering time |
|:---------------:|:--------:|:------------------:|
|  original model |   26.67  |        3.76        |
| attention model |   26.91  |        61.87       |

### Training Records
