## Installation steps:
`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

### Then install:
`randaugment` <br>
`pycocotools` <br>
`chardet` <br>

## Further changes:
* We change the `np.int` in randaugment to `np.int32`
* We comment out the redundant `symbol` library import
* We disable the dependency on the `inplace_abn` library