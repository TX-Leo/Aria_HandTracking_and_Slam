 xvfb-run python aria.py --mps_path "./test_data/mps_open_door_5_vrs/"

# The folder should be like:
# - test_data
#     - mps_TEST_vrs/
#         - else
#             - sample.vrs.json
#             - vrs_health_check.json
#             - vrs_health_check_slam.json
#         - hand_tracking
#             - hand_tracking_results.csv
#             - summary.json
#         - slam
#             - closed_loop_trajectory.csv
#             - online_calibration.jsonl
#             - open_loop_trajectory.csv
#             - semidense_observation.csv.gz
#             - semidense_points.csv.gz
#             - summary.json
#         - sample.vrs
# 