# Install Aria Mobile App
Follow [ARK SW Downloads and Updates](https://facebookresearch.github.io/projectaria_tools/docs/ARK/mobile_companion_app). Install the app, sign in and pair.

# Add Environment Variables
```
vim ~/.bashrc
```

Add the following environment variables to your ~/.bashrc or ~/.zshrc with your institution's username and password

```
export ARIA_MPS_UNAME="your_uname"
export ARIA_MPS_PASSW="your_passw"
```

```
source ~/.bashrc
```

# Install Aria Studio
```
pip install aria_studio --no-cache-dir
```

test it:
```
aria_studio
```


# Verify that your dependencies have been installed correctly.
```
aria-doctor
```

You should see:
```
[ pass] glibc version ok
[ pass] Python version ok
[ pass] Aria udev rules ok
[ pass] Aria network manager connection ok
```

# Pair the glasses via USB to your computer. Verify that you have connected correctly.
```
aria auth pair
```
You should see:
```
[AriaCli:App][INFO]: Attempting to send authentication pairing request to device over USB. Please ensure the device is connected to a USB port.
[AriaCli:App][INFO]: Sent authentication request with hash xxx to device. Please check and approve the request in the companion app.
```

# Try to record via the phone
You could customize the profile you want. We recommend you choosing the ```Profile 10```.

After recording, you should connect the glasses to your computer and run ```aria_studio```, download the data you just recorded to ```./test_data/```, which should be a ```.vrs``` file. 

# Install Projectaria Tools and Client SDK
```
pip install projectaria_client_sdk==1.1.0 --no-cache-dir
pip install projectaria_tools==1.7.1
```

# Data processing on the MPS server
Submit the video for data processing on the MPS server and reorganize the output folder. Job submission may take anywhere from 5 to 30 minutes. For example, if your ```.vrs``` file is ```TEST.vrs```, you would run

```
conda activate aria
cd src/data/
NAME="open_cabinet_5"
aria_mps single --force -i "${NAME}.vrs" -u "$ARIA_MPS_UNAME" -p "$ARIA_MPS_PASSW" --features SLAM HAND_TRACKING
NAME="open_cabinet_5"
mv "${NAME}.vrs" "mps_${NAME}_vrs/sample.vrs"
mkdir -p "mps_${NAME}_vrs/else/"
mv "${NAME}.vrs.json" "mps_${NAME}_vrs/else/sample.vrs.json"
mv "mps_${NAME}_vrs/vrs_health_check.json" "mps_${NAME}_vrs/else/vrs_health_check.json"
mv "mps_${NAME}_vrs/vrs_health_check_slam.json" "mps_${NAME}_vrs/else/vrs_health_check_slam.json"
```

After this, it should be:
```
- test_data
    - mps_NAME_vrs/
        - else
            - sample.vrs.json
            - vrs_health_check.json
            - vrs_health_check_slam.json
        - hand_tracking
            - hand_tracking_results.csv
            - summary.json
        - slam
            - closed_loop_trajectory.csv
            - online_calibration.jsonl
            - open_loop_trajectory.csv
            - semidense_observation.csv.gz
            - semidense_points.csv.gz
            - summary.json
        - sample.vrs
```

# Visualize the aria sensors
```
viewer_aria_sensors --vrs "./test_data/mps_NAME_vrs/sample.vrs"
```

# Visualize the hand tracking and slam
```
viewer_mps --vrs "./test_data/mps_TEST_vrs/sample.vrs" \
--trajectory "./test_data/mps_TEST_vrs/slam/closed_loop_trajectory.csv" \
--points "./test_data/mps_TEST_vrs//slam/semidense_points.csv.gz" \
--hands_all "./test_data/mps_TEST_vrs/hand_tracking/hand_tracking_results.csv" \
--web
```