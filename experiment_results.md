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

#### original model

* PSNR
<img src="https://raw.githubusercontent.com/FerdinandZhong/nerf-pytorch/master/experiments_charts/original/training%20PSNR%20(steps).svg" width="75%">

* loss
<img src="https://raw.githubusercontent.com/FerdinandZhong/nerf-pytorch/master/experiments_charts/original/training%20loss%20(steps).svg" width="75%">

* learning rate
<img src="https://raw.githubusercontent.com/FerdinandZhong/nerf-pytorch/master/experiments_charts/original/training%20lr%20(steps).svg" width="75%">

#### attention model

* PSNR
<img src="https://raw.githubusercontent.com/FerdinandZhong/nerf-pytorch/master/experiments_charts/attention/training%20PSNR%20(steps).svg" width="75%">

* loss
<img src="https://raw.githubusercontent.com/FerdinandZhong/nerf-pytorch/master/experiments_charts/attention/training%20loss%20(steps).svg" width="75%">

* learning rate
<img src="https://raw.githubusercontent.com/FerdinandZhong/nerf-pytorch/master/experiments_charts/attention/training%20lr%20(steps).svg" width="75%">
