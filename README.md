# Plantar Pressure Prediction

This module contains code in support of 
### Dependencies

The code was successfully built and run with these versions:

```
Python 3.6
toch 1.10.0-cu113
```
Note: You can also create the environment I've tested with by importing _environment.yml_ in conda.

### Data
You can download the data for [healthy people](https://livewarwickac-my.sharepoint.com/:u:/r/personal/u1880714_live_warwick_ac_uk/Documents/Pred_Plantar/CAD_WALK_Healthy_Controls_Dataset.zip?csf=1&web=1&e=gDThuY) and [unhealthy people](https://livewarwickac-my.sharepoint.com/:u:/r/personal/u1880714_live_warwick_ac_uk/Documents/Pred_Plantar/CAD_WALK_Hallux_Valgus_PreSurgery.zip?csf=1&web=1&e=1Iv7IV) by clicking on them.

You can replicate the results with putting the data in a similar directory tree as below:

```
/data/
  coco/
     CAD_WALK_Hallux_Valgus_PreSurgery/
        HalluxValgus_PreSugery
            HV01/
                ...
     CAD_WALK_Healthy_Controls_Dataset/
        HealthyControls
            HC01/
                ...
```

### References

If you found this repo useful give me a star!