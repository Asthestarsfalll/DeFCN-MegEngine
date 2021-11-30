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
mkdir data
ln -s /path/to/your/dataset data
```

- train

```sh
# check scripts folder to custom
sh scripts/poto_coco_800size_3dmf_aux_gn.sh
```

- test

```sh
sh scripts/test.sh
```

- inference

```sh
python inference.py --help
python inference.py -w './data' -i './xx.jpg'
```

