# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.


## Table of Contents
1. [Installation](#installation)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Results](#results)

### Installation <a name="installation"></a>
For running this project, the most important library is Python version of Anaconda Distribution. It installs all necessary packages for analysis and building models. 


### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        ` python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run 'python run.py'
3. Open another terminal and type 'env|grep WORK'
   this will give you the spaceid it will start with view*** and some characters after that
4. Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id you got in the step 3
5. Press enter and the app should now run for you

Local Machine

1. Run 'python run.py'
2. Go to to localhost:3001 and app will run



### File Descriptions <a name="files"></a>
1. data/process_data.py: The ETL pipeline used to process and clean data in preparation for model building.
2. models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle.
3. app/templates/*.html: HTML templates required for the web app.
4. app/run.py: To start the Python server for the web app and render visualizations.

### Results<a name="results"></a>
The main observations of the trained classifier can be seen by running this application.

