# Aria Usage
About how to use the Meta Aria Gen1 Glasses and MPS to collect data, you should check [README_aria_data_collection](./README_aria_data_collection.md)

# Setup
```
conda create -n aria python=3.10
conda activate aria
pip install -r requirements.txt
```
download devignetting_masks:
```
curl -L -o devignetting_masks.zip "https://www.projectaria.com/async/sample/download/?bucket=core&filename=devignetting_masks_bin.zip"
unzip -o ./devignetting_masks.zip -d ./devignetting_masks
rm ./devignetting_masks.zip
```
download a test data:
```
TODO
```
# Run aria
```
bash run_aria.sh
```

The results are in ```./test_data/mps_TEST_vrs/aria/```

