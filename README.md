# MGNN-SPred
- This is our Tensorflow implementation for the paper:
  > WenWang,Wei Zhang, Shukai Liu, Qi Liu, Bo Zhang, Leyu Lin, and Hongyuan Zha. 2020. Beyond Clicks: Modeling Multi-Relational Item Graph for Session-Based Target Behavior Prediction. In Proceedings of The Web Conference 2020 (WWW ’20), April 20–24, 2020, Taipei, Taiwan.


- A deep learning model for session-based target behavior prediction

## Support
- Python version: 3.6.9
- tensorflow version: 1.12.0

# Dataset
- https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z

## Usage:
### data:
- ./run_time/data/yc/yoochoose-buys.dat
- ./run_time/data/yc/yoochoose-clicks.dat

### command
- python3 preprocessing_data.py
- python3 main.py
