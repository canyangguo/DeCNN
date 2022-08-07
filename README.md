## De-correlation Neural Network for Synchronous Implementation of Estimation and Secrecy

## run
```
python main.py --alpha 0 --gamma 10000 --model_name 'LSTM' --random_seed 0 --epochs 10000 --gpu 0
```

## dataset
```
--Lat: latitude
--Lon: longitude
--c: RSSI from base station
--w: RSSI from Wi-Fi access points
```

## requirements
* python 3.7.10
* torch 1.9.0
* numpy 1.20.2
* pandas 1.2.5