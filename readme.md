The MegEngine implement of DeFCN.

## Get Started

```sh
git clone https://github.com/Asthestarsfalll/DeFCN-MegEngine.git
pip3 install megengine -f https://megengine.org.cn/whl/mge.html
pip install -r requirements.txt
```

- prepare datasets

```sh
cd DeFCN-MegEngine
# the dataset path should contain coco or crowdhuman
ln -s /path/to/your/dataset data
```

​	coco dataset expect the folder format

```sh
|- coco
 |-  annotations
 |-  test2017
 |-  train2017
 |-  val2017
```

​	crowdhuman dataset expect  the folder  format

```sh
|- crowdhuman
 |- Images #contain all train and test images
 |- annotation_train.json
 |- annotation_val.json
```

​	To get `annotation json file`,  you can run `crowdhuman2coco.py` to convert odgt files

```sh
python crowdhuman2coco.py --help
python crowdhuman2coco.py -d /path/to/crowdhuman/dataset -o /path/to/odgt/file -j /path/to/save
```

- train

```sh
# check scripts folder to custom
sh scripts/poto_coco_800size_3dmf_aux_gn.sh
```

- test

```sh
# check scripts folder to custom
python test.py --help
sh scripts/test.sh
```

- inference

```sh
python inference.py --help
python inference.py -w './data' -i './xx.jpg'
```

