# Injury Report Analysis for Seattle Bouldering Project  
Liz Berg | Galvanize, Inc. | The Seattle Bouldering Project | January 2018

### Motivation: 
The [Seattle Bouldering Project](seattlboulderingproject.com) is currently allocating energy and resources toward a broad goal of reducing the rate of climbing-related injuries in the gym. SBP hopes to gain actionable insights that may affect the structure of:
* The content of safety information that is presented to visitors, and the method in which it is given
* The priorities and attention of employees during their shifts, particularly during safety walks & facility checks
* The routes that are set (specific holds, movements, climbing styles) . 

To address these issues, random forest and gradient boosted models were created to predict the probability of injury on a given visit to the gym. Key features from the models were then extracted and analyzed to identify pertinent risk factors and make recommendations for injury prevention strategies. 

---

### Questions:
#### 1. Who is getting injured?
#### 2. When do injuries tend to occur?
#### 3. Are the prior assumptions that gym employees have about injuries backed up by data?
#### 4. Are any contributing factors under the control of SBP?

---

### The Data: 
![alt text](https://github.com/elisabeth-berg/sbp-injury-public/blob/master/img/data.png)

* Incident reports from 2016 and 2017: Qualitative data (name, date, description, location, etc.) about every injury that occurred within the gym, handwritten and then typed into a spreadsheet. 
   
* User check-ins from 2011-2017: A record of every visit to the gym since it opened. 
   
* Employee timesheets from 2016 and 2017: Clock-in and clock-out times for every employee. 

The raw data described above contains user-sensitive material such as name and birthdate, so is not included in this repository.  

### Final Dataset: 
The raw data was heavily cleaned, refactored, and merged to create a dataframe in which each row represents a single visit to the gym. 
Several features were also engineered from the raw check-in data in order to capture user behavior and experience level. The final data used for the model is contained in `data/df_formodel.pkl` and contains the following features:  

|      |                                   |
|------|-----------------------------------|
| hour | The hour that the user checked in |
| weekday | 7 dummy variables, one for each day of the week |
| new_set | Boolean, whether a new wall was set on this day before the time of checkin |
| age | Current age of the visitor |
| occupancy | The total number of people in the gym, under the assumption that each person climbs for 2 hours |
| num_employes | The number of front desk & youth programs employees clocked in at the given day and hour |
| user_type | 3 dummy variables: 'member', 'punch_card', and 'guest' |
| longevity | The number of days since the user's first visit |
| visit_count | The total number of times this user has been to the gym previously |
| visits_3 | This user's number of visits in the last 3 days |
| visits_7 | This user's number of visits in the last 7 days |
| visits_14 | This user's number of visits in the last 14 days |
| visits_30 | This user's number of visits in the last 30 days |


### The Models: 

| Gradient Boost | Random Forest |
|----------------|---------------|
|`max_depth = 2` | `max_depth = 6`|
|`learning_rate = 0.005` | `min_samples_split = 2` | 
|`n_estimators = 1150` |   `n_estimators = 1000`|
|`subsample = 0.5`|

![alt text](https://github.com/elisabeth-berg/sbp-injury-public/blob/master/img/roc_compare.png)



Both models are able to pick up on subtle distinctions between visits that result in injury and those that do not. 
Since the point of this research is to identify features that correlate with injury occurence, we would like to investigate how the model draws these distinctions.  

The partial dependency plots shown below are generated from the random forest model, along with 100 models fit to bootstrapped data. The gradient boosted model demonstrated similar partial dependencies. 

![alt text](https://github.com/elisabeth-berg/sbp-injury-public/blob/master/img/pd_age.png) 
![alt text](https://github.com/elisabeth-berg/sbp-injury-public/blob/master/img/pd_longevity.png)
![alt text](https://github.com/elisabeth-berg/sbp-injury-public/blob/master/img/pd_occupancy.png)
![alt text](https://github.com/elisabeth-berg/sbp-injury-public/blob/master/img/pd_visit_count.png)
![alt text](https://github.com/elisabeth-berg/sbp-injury-public/blob/master/img/pd_hour.png)
![alt text](https://github.com/elisabeth-berg/sbp-injury-public/blob/master/img/pd_visits_30.png)
![alt text](https://github.com/elisabeth-berg/sbp-injury-public/blob/master/img/pd_visits_14.png)

### Conclusions & Insights: 
#### 1. First time users are particularly injury-prone. Additional measures should be taken to educate and support new visitors. 

#### 2. Emphasize safety instruction for youth -- even those who have been climbing for a long time.

#### 3. Late hours tend to result in more injuries, regardless of the gym occupancy. Staff should be encouraged to remain attentive, even as the night winds down and the gym is less busy. 



### How to Run This:

`import pandas as pd`  
`import pickle`  
`from src import models`  


Load the full dataframe with features:  
`df = pd.read_pickle('data/df_formodel.pkl')`  
`X = df.drop(columns=['injured'])`    
`y = df['injured']` 

Fit a model:  
`model_pkl = open('rf_model.pkl', 'rb')`  (for Random Forest)  
`model_pkl = open('gb_model.pkl', 'rb')`  (for Gradient Boost)   
`model = pickle.load(model_pkl)`

Predict the probability of injury for a given visit:  
`"""`  
`feature_values : a numpy array of values corresponding to the following features`  
` ['hour', 'new_set', 'age', 'visit_count', 'occupancy', 'longevity',  
       'num_employees', 'guest', 'member', 'punch', 'mon', 'tues', 'wed',  
       'thurs', 'fri', 'sat', 'sun', 'visits_3', 'visits_7', 'visits_14', 'visits_30']`  
` """`  
`one_visit = feature_values.reshape(1, -1)`  
`injury_prob = model.predict(one_visit)`  

Note that automatic extraction of checkin features requires access to sensitive user information (name, birthdate), so you must input these values manually. Future work will anonymize each visitor by assigning a unique ID. This will enable public deployment of the feature engineering pipeline. 
