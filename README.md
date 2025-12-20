This is a plug-and-play python3 module, which process the hand_tracking and slam data from Meta Aria Gen1 Glasses MPS. 

Hand tracking: visualization of hand keypoints, palm&wrist pose, and semidense point clould. 

Slam: Optimize the human's movement trajectory, phase estimation, and estimation the linear and angular velocity.

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
unzip -o ./devignetting_masks.zip -d ./aria_devignetting_masks
rm ./devignetting_masks.zip
```
download a test data:
```
TODO Should upload the mps_TEST_vrs to the cloud
```
# Run aria
```
bash run_aria.sh
```

The results are in ```./test_data/mps_TEST_vrs/aria/```

# One Example
Hand Tracking:

![Door Opening](https://github.com/user-attachments/assets/8572ddb3-c016-4e16-afd3-f2848cb75728)

Slam:
