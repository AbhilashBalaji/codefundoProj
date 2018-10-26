# PREDICTING HURRICANES AND DEVELOPING A GENERALIZED RESCUE OPTIMISATION MODEL



## STAGE 1:PREDICTION OF THE HURRICANES 
Using NOAA data archives for past hurricane and hurricane-like events, such as location pressure data, wind flow and various other factors,we then acquire useful features using various ML and Deep learning algorithms. 

Then we acquire live NOAA and BOA data using various WEB APIs and then use the previously acquired features and compare it with the current data to predict the possibility of a hurricane, mainly by monitoring certain hot-spots in a hurricane prone region for any anomaly in the live data. 
If any anomaly is noticed, then the region is flagged and continuously monitored and data fed to the respective authorities until it is definitive that a hurricane is going to arise. 

## STAGE 2: OPTIMISATION OF EVACUATION OPERATION
Once stage 1 is executed optimally, Stage 2 begins. Here, a case study of a particular city will be taken into consideration for better explanation.Road mapping and population density will be the main factors will be taken into account as the main criteria. These factors will help evacuate the civilians in the fastest and the most efficient way possible. Also, the live data will be continuously monitored and the location of subsidence of the hurricane and the max destruction radius will be calculated and continuously be informed to the civilians. 

Further more, an application can be implemented and open sourced, which on installation on the civilian's phone can be used to give live updates to the user about the change in direction of the hurricane and also the speed and intensity.

## [Enter Site](https://naturaldisasterprediction.azurewebsites.net)
## NOTE : Model will not work if website supports same origin policy ,as CORS is not suported by AzureML
[how to disable same origin policy](https://stackoverflow.com/questions/3102819/disable-same-origin-policy-in-chrome#6083677)

### Scrapping Data:

```bash
python ./Model/SCRAPINGALGO.py
```

### Extracting Model:

```bash
python ./Model/labeler.py
```
### Running Model Server:

```bash
python Model_server.py
```

### MODEL Specifications 
Gaussian Navieve Bayes estimator 
avg Acc : 93%  with 33% Train data 

## Services used 
* Azure Database for MYSQL server
* ML service workspace
* Application Insights
* Virtual Networks
* Lots of VMs
* AzureML Webservices 








