# DisasterResponsePipeline
Analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages.
The user may then enter a message to the web app to classify the message based on the model.
## Running the project
This project was created mainly using the terminal to run the application. If using an IDE that builds the run command for
you, make sure the working director is set to {PATH}\DisasterResponsePipeline.<br>
<br>
When running through the terminal use these following commands.
1. cd into DisasterResponsePipeline
1. pip install .
1. python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
1. python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
1. cd into the app folder
    1. python run.py
    1. Navigate to the website via web browser. Use the following commands to determine the URL if in the udacity workspace:
        1. env|grep WORK (in a new terminal)
        1. From the above command build the url as such. Where SPACEID AND SPACEDOMAIN are the values shown in the previous
       command. 
            1. https://SPACEID-3001.SPACEDOMAIN
    1. If on local, simply use http://localhost:3001

## File Descriptions - DisasterResponsePipeline
### /data
* process_data.py - Uses the ETL class to perform extraction, transformation, and load the data
* disaster_categories.csv - contains all the categories of each message
* disaster_messages.csv - contains all the messages
* DisasterResponse.db - Name may change on user input, but this is the resulting database created after the ETL process. 

### /etl
* etl.py - Contains the class which extracts, transforms, and loads data into an in memory database from csv files
  located in the data folder.
### /models 
* train_classifier.py - This file is responsible for creating a model, testing it, and displaying the results.
* classifier.pkl - The resulting model built and used for future classification.