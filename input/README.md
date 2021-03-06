# Input folder

Here lies the input for the simulator. It contains the following files:

+ consumer_event.csv
Contains a list of consumer events. Simulates real world usage.
Column description:
  + When the load was registered in the system (UNIX timestamp)
  + Earliest Start Time (EST) (UNIX timestamp)
  + Latest Start Time (LST) (UNIX timestamp)
  + ID of the device [houseId]:[deviceId]:[id]
  + The name of the csv file that contains the load profile

+ producer_event.csv
Contains a list of prediction updates for the solar panels.
Column description:
  + Timestamp (UNIX)
  + ID of the panel [houseId]:[producerId]
  + The name of the csv file that contains the prediction

+ loads
Contains the load profile files. Files with the 'back' name represents background loads, which are inflexible loads that can't be shifted and has a strict start time. (For example lights) 
Column description:
  + Seconds elapsed since start
  + Cumulative power usage (Wh)

+ predictions
Contains the predicted power generated by the solar panels.
File names: [houseId] [panelId] [Which prediction for the day]
Colum description:
  + Timestamp (UNIX)
  + Cumulative power production that day (Wh)
