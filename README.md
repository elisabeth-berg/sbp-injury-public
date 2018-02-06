# Injury Report Analysis for Seattle Bouldering Project  
Liz Berg | Galvanize, Inc. | The Seattle Bouldering Project | January 2018

### Summary: 
The [Seattle Bouldering Project](seattlboulderingproject.com) is currently allocating energy and resources toward a broad goal of reducing the rate of climbing-related injuries in the gym. SBP hopes to gain actionable insights that may affect the structure of:
* The content of safety information that is presented to visitors, and the method in which it is given
* The priorities and attention of employees during their shifts, particularly during safety walks & facility checks
* The routes that are set (specific holds, movements, climbing styles) . 

To address these issues, random forest and gradient boosted models were created to predict the probability of injury on a given visit to the gym. Features from the models were then extracted and analyzed to identify key risk factors and make recommendations for injury prevention strategies. 


### Questions:
#### 1. Who is getting injured?
#### 2. When do injuries tend to occur?
#### 3. Are the prior assumptions that gym employees have about injuries backed up by data?
#### 4. Are any contributing factors under the control of SBP?

---

### The Data: 
![alt text](https://github.com/elisabeth-berg/sbp-injury/blob/master/img/data.png)


* Incident reports from 2016 and 2017: Qualitative data about every injury that occurred within the gym.  

   `| name | birthdate | date | time | description | injury | location | color | 911 |`  
   
     &nbsp;&nbsp;&nbsp; `description` : an unstructured description of the incident.  
     &nbsp;&nbsp;&nbsp; `injury` : bodily location of the injury.   
     &nbsp;&nbsp;&nbsp; `location` : area of the gym in which the incident occurred.  
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `color` : difficulty level of the climb.  
     &nbsp;&nbsp;&nbsp; `911` : boolean, True if 911 was called in response to the incident.  
   
* User check-ins from 2011-2017: A record of every visit to the gym since it opened. 

   `| date | hour | name | age_as_of_report_end | user_type |`  
   
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `age_as_of_report_end` : age on the date that the report was generated (2017-12-22).  
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `user_type` : membership status, one of `'MEMBER'`, `'PUNCH CARD'`, or `'GUEST'`.
   
* Employee timesheets from 2016 and 2017: Clock-in and clock-out times for every employee. 

### Final Dataset: 
Several features were engineered from the raw check-in data in order to capture user behavior and experience level. Relevent data from the incident reports and the employee timesheets were also merged into a final dataframe with the following features:  
 `hour | weekday | new_set | occupancy | num_employees | user_type | longevity | visit_count | visits_3 | visits_7 | visits_14 | visits_30` 


### The Models: 
![alt text](https://github.com/elisabeth-berg/sbp-injury/blob/master/img/roc_compare.png)

| Gradient Boost | Random Forest |
|----------------|---------------|
|`max_depth = 2` | `max_depth = 5`|
|`learning_rate = 0.005` | `min_samples_split = 2` | 
|`n_estimators = 1150` |   `n_estimators = 1000`|
|`subsample = 0.5`|



### Conclusions & Actionable Insights: 
#### 1. First time users are much more injury-prone. Additional measures should be taken to educate and support new visitors. 
#### 2. Emphasize safety instruction for youth -- even those who have been climbing for a long time. 
#### 3. Late hours tend to result in more injuries, regardless of the gym occupancy. Staff must remain attentive, even as the night winds down and the gym is less busy. 


### How to Run This:
Load the full dataframe with features:  
`s3 = boto3.client('s3')`  
`pickled_data = s3.get_object(Bucket='sbp-data-etc', Key='df_formodel.pkl')`  
`pickle_file = BytesIO(pickled_data['Body'].read())`  
`df = pd.read_pickle(pickle_file)`  
`X = df.drop(columns=['injured'])`    
`y = df['injured']` 
