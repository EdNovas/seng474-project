# <u>Project Report</u>

## SENG 474 Data Mining Project Report

### Historical Data Mining for Predicting Air Quality in Victoria, Canada

Alvin Guo(V00987315)<br>
Daming Wang(V00960801)<br>
Christopher Xu(V01007912)<br>
<br>
<br>

## Table of Contents

1. [Introduction]()
2. [Research Methodology]()
3. [Data Mining]()
4. [Expected Findings]()
5. [Implications and Future Work]()
6. [Conclusion]()

## 1.0 Introduction

### 1.1. Background<br>

Air quality has emerged as a crucial factor influencing not only the environment but also the health and well-being of populations worldwide. In recent years, the city of Victoria, Canada, like many urban environments, has been grappling with complex air quality challenges. These challenges stem from a confluence of factors including climate change, industry emissions, vehicular pollutants, and rapidly evolving urban development patterns. All these elements interact in complex ways, altering the constituents of the air and consequently, its quality.

Research has demonstrated a significant correlation between poor air quality and adverse health outcomes such as respiratory diseases, cardiovascular complications, and premature deaths. Further, the environment bears the brunt of pollution, with implications for biodiversity, weather patterns, and climate change. Therefore, understanding and predicting air quality is paramount in planning for healthier and more sustainable urban environments.

In this light, the present research seeks to explore a novel approach to predicting air quality - using data mining techniques on historical data. While this area remains largely uncharted, especially in the context of Victoria, the potential benefits for policy-making, public health planning, and environmental conservation are immense.

### 1.2. Objective

The primary objective of this research is to develop a data-driven model capable of accurately predicting future air quality in Victoria, Canada, based on historical data. To meet this objective, the study is guided by the hypothesis that patterns and trends in past air quality data, when combined with relevant meteorological and anthropogenic factors, can provide reliable predictive capabilities.

The overarching goal of this research is not only to contribute to the scientific understanding of air quality dynamics but also to provide actionable insights that can guide policy development, inform public health strategies, and support environmental conservation efforts in Victoria and beyond.

<br>
<br>

## 2.0 Research Methodology
### 2.1. Data Collection

We will use historical air quality data from Victoria's monitoring stations and government reports, spanning $several$ years. Additional data will be obtained from the meteorological department, including parameters such as $temperature$, $humidity$, and $pressure$. We will also collect data on anthropogenic factors, such as population density, traffic, and industrial activities.

### 2.2. Data Preprocessing

#### 2.2.1. Raw Database:
##### For Windows Operating System:
a. Open File Explorer.
b. Click on the folder icon located on the taskbar.
Alternatively, press the Windows key + E on your keyboard.
Enter the URL for FTP:
c. Locate the File Explorer address bar at the top of the window.
Type the desired URL into the address bar. For example, "ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/AnnualSummary/" is the URL for our project.
<br>
<br>
##### For UNIX/LINUX Operating System:
a. Open the terminal.<br>
b. Type the desired URL into the address bar. For example, "ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/AnnualSummary/" is the URL for our project.
<br>
<br>
#### 2.2.2. Initiate the Connection:
a. Press the Enter key or click the Go button situated next to the address bar.

b. File Explorer will attempt to establish a connection with the provided FTP serve.


#### 2.2.3. Verify the Connection:
a. Upon successful connection, the contents of the FTP server will be displayed in the File Explorer window.

b. Users can now navigate through the directories, open files, and perform various file operations.

#### 2.2.4. Data Combine:
The way we used to combine multiple CSV files containing PM2.5
monitoring data from 2011 to 2021 into a single file named
"PM25_2010_to_2020.csv".:

1. Importing Required Libraries

- The code begins by importing the necessary libraries: `pandas` and `google.colab.drive'.
- `pandas` is used for data manipulation and analysis.
- `google.colab.drive` is used to mount Google Drive in order to access the data files.

2. Mounting Google Drive:
- The line `drive.mount('/content/drive')` mounts the Google Drive to the Colab notebook. This allows
access to the files stored in Google Drive

3. Setting the Directory:
- The line `os.chdir("/content/drive/MyDrive/Seng474Project/PM25")` changes the current working
directory to the specified path. Make sure to provide the correct path where PM2.5 data files are located.

4. Defining File Extension:
- The line `extension = 'csv'` sets the file extension variable to 'csv'. This specifies that we want to work with CSV files.

5. Getting File Names:
- The line `all_filenames = [i for i in glob.glob('*.{}'.format(extension))]` uses the `glob.glob` function to
retrieve a list of all file names in the current directory that match the specified file extension ('csv' in this
case). It searches for all CSV files in the directory and stores their names in the `all_filenames` list.

6. Combining CSV Files:
- The line `combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])` reads each CSV file from
the `all_filenames` list using a list comprehension and concatenates them into a single DataFrame using
`pd.concat()`. Each CSV file is read as a separate DataFrame using `pd.read_csv()`, and all the DataFrames
are combined into one using `pd.concat()`.

7. Saving Combined Data to CSV:
- The line `combined_csv.to_csv( "PM25_2010_to_2020.csv", index=False)` saves the combined DataFrame as a new CSV file named "PM25_2010_to_2020.csv" in the current directory. The `index=False` parameter specifies not to include the index column in the output file.

- Make sure to adjust the directory path and file extension based on specific setup. Once executed, this code will combine all the PM2.5 monitoring data files from 2011 to 2021 into a single CSV file named "PM25_2010_to_2020.csv" in the specified directory.

- The code demonstrates a data management process for PM2.5 values from 2010 to 2020, as well as for the year 2021. It utilizes the `os` and `gdown` libraries in Python for file management and downloading from Google Drive.

#### 2.2.5 Data Process Broken Down

The data process can be broken down into the following steps:

1. Checking File Existence:<br>
The code begins by defining a function named `file_exists()` that checks if a file exists locally. This function is later used to determine whether the required data files are already present or need to be downloaded.

2. Retrieving PM2.5 Data from 2010 to 2020:<br>
The code utilizes a Google Drive sharing link to access the PM2.5 data from 2010 to 2020. It extracts the file ID from the sharing link and defines the output file path for storing the downloaded data. It then checks if the local file exists using the `file_exists()` function. If the file is not found, it proceeds to download the CSV file containing the PM2.5 data using the `gdown.download()` function. Otherwise, it simply prints a message indicating the use of the local file.

3. Retrieving PM2.5 Data for 2021:<br>
Similarly, the code defines a Google Drive sharing link for the PM2.5 data in 2021 and extracts the file ID. It defines the output file path for storing the data and checks if the local file exists. If the file is not found, it downloads the CSV file using the `gdown.download()` function. Otherwise, it prints a message indicating the use of the local file.<br>
Overall, this code segment focuses on retrieving and managing PM2.5 data for the specified years. It ensures that the necessary data files are available either by downloading them from Google Drive or using the local copies if they already exist. This data preparation step is essential for subsequent analysis and modeling tasks, such as PM2.5 value prediction and forecasting.

4. Importing Libraries:<br>
The code starts by importing necessary libraries for data analysis and modeling, including `pandas`, `sklearn` modules, `matplotlib.pyplot`, `numpy`, `statsmodels.tsa.arima.model`, and `datetime`.

5. Loading PM2.5 Training Data:<br>
The code uses the `pd.read_csv()` function from the `pandas` library to read the PM2.5 training data from the file '/content/pm25_2010_to_2020.csv' into a pandas DataFrame named `PM25_train`. This dataset contains historical PM2.5 values from 2010 to 2020. The `head()` function is then called to display the first few rows of the dataset.

6. The `train_and_predict()` Function:<br>
The `train_and_predict` function takes four parameters:
  - `model`: The machine learning model to be trained and used for prediction.
  - `X_train`: The input features of the training data.
  - `y_train`: The target variable (PM2.5 values) of the training data.
  - `X_test`: The input features of the test data.<br><br>
Within the function:
  - The `model.fit(X_train, y_train)` line trains the specified model using the training data, where `X_train` represents the input features and `y_train` represents the corresponding target variable.
  - The trained model is then used to make predictions on the test data using the `model.predict(X_test)`
line.
  - The predicted values are returned as the output of the function.
This function encapsulates the process of training a given model with the provided training data and using it to make predictions on the test data. It abstracts away the specific training and prediction logic, making it convenient to use different models interchangeably for prediction tasks.

7. `evaluate_model()` Function:<br>
The `evaluate_model` function takes two parameters:
  - `y_test`: The actual target variable values from the test data.
  - `y_pred`: The predicted target variable values<br><br>
Within the function:
  - The `mean_squared_error(y_test, y_pred)` function from the `sklearn.metrics` module is used to calculate the MSE between the actual and predicted values.
  - The `mean_absolute_error(y_test, y_pred)` function from the same module is used to calculate the MAE between the actual and predicted values.
  - The calculated MSE and MAE values are then returned as a tuple `(mse, mae)`

8. Calculating Mean PM2.5 Value:
  - `PM25_train['RAW_VALUE']` represents the 'RAW_VALUE' column in the `PM25_train` DataFrame, which contains the PM2.5 values.
  - The `.mean()` function is applied to this column to calculate the average or mean value of the PM2.5 readings from the training data.<br><br>
This calculation provides the mean PM2.5 value from the available training data. It can be used to gain insights into the average level of PM2.5 pollution during the specified time period (2010 to 2020).

9. `preprocess_data()` Function:
The `preprocess_data` function takes a DataFrame (`df`) as a parameter and performs the following
preprocessing steps:
  - Converting Date and Time.
    - The 'DATE' column in the DataFrame is converted to a datetime format using `pd.to_datetime(df['DATE'])`.
    - The 'YEAR', 'MONTH', 'DAY', and 'HOUR' columns are created by extracting the corresponding values from the 'DATE' and 'TIME' columns using the `dt.year`, `dt.month`, `dt.day`, and `apply(lambda x: int(x.split(':')[0]))` operations, respectively.
  - Dropping Columns:
    - The 'DATE', 'DATE_PST', 'TIME', and 'ROUNDED_VALUE' columns are dropped using `df.drop(['DATE', 'DATE_PST', 'TIME','ROUNDED_VALUE'], axis=1)`.

  - Encoding Categorical Columns:
    - The 'REGION' and 'STATION_NAME' columns are encoded using label encoding. The `LabelEncoder()` object is created as `le`, and then `le.fit_transform()` is applied to these columns (`df['REGION']` and `df['STATION_NAME']`).
  - Dropping Additional Columns:
    - Several additional columns such as 'STATION_NAME_FULL', 'NAPS_ID', 'UNIT', 'INSTRUMENT', 'OWNER', 'EMS_ID', and 'PARAMETER' are dropped using `df.drop(['STATION_NAME_FULL', 'NAPS_ID', 'UNIT', 'INSTRUMENT', 'OWNER', 'EMS_ID', 'PARAMETER'], axis=1)`.
  - Handling Missing Values:
    - The missing values in the DataFrame are filled with the mean value of each column using `df.fillna(df.mean())`.
  - Returning the Preprocessed DataFrame:
    - The preprocessed DataFrame is returned as the output of the function.
This function prepares the data for further analysis and modelling by converting date and time columns,
encoding categorical variables, dropping unnecessary columns, and filling missing values with the mean.
These preprocessing steps help to ensure that the data is in a suitable format for modelling purposes.

10. Linear Regression:
  - A `LinearRegression` model (`lr`) is instantiated.
  - The `train_and_predict()` function is called with the Linear Regression model, training data (`X_train`, `y_train`), and test data (`X_test`) to make predictions (`y_pred_lr`).
  - The MSE and MAE between the actual test labels (`y_test`) and the predicted labels (`y_pred_lr`) are calculated and stored in `mse_lr` and `mae_lr`, respectively

11. Logistic Regression:
  - Binary classification is performed by converting the target variable `y` into a binary variable based on a threshold value (`threshold`). In this case, whether the 'RAW_VALUE' is above the mean value of the training data.
  - The Logistic Regression model (`logr`) is instantiated.
  - The Logistic Regression model is trained using the training data (`X_train_log`, `y_train_log`).
  - The model is used to make predictions (`y_pred_logr`) on the test data (`X_test_log`).
  - The MSE and MAE between the actual test labels (`y_test`) and the predicted labels (`y_pred_logr`) are
calculated and stored in `mse_logr` and `mae_logr`, respectively.
  - The accuracy of the Logistic Regression model is calculated using `accuracy_score` between the actual
binary labels (`y_test_log`) and the predicted binary labels (`y_pred_logr`).
12. Decision Tree:
  - A `DecisionTreeRegressor` model (`dt`) is instantiated.
  - The `train_and_predict()` function is called with the Decision Tree model, training data (`X_train`,
`y_train`), and test data (`X_test`) to make predictions (`y_pred_dt`).
  - The MSE and MAE between the actual test labels (`y_test`) and the predicted labels (`y_pred_dt`) are calculated and stored in `mse_dt` and `mae_dt`, respectively.
13. Random Forest:
  - A `RandomForestRegressor` model (`rf`) is initiated with specified hytperparameters. 
14. Gradient Boosting:
  - A `GradientBoostingRegressor` model (`gb`) is instantiated with specified hyperparameters.
  - The `train_and_predict()` function is called with the Gradient Boosting model, training data (`X_train`,
`y_train`), and test data (`X_test`) to make predictions (`y_pred_gb`).
  - The MSE and MAE between the actual test labels (`y_test`) and the predicted labels (`y_pred_gb`) are calculated and stored in `mse_gb` and `mae_gb`, respectively.
15. Printing the Errors:
  - The calculated MSE and MAE values for each model are printed to evaluate their performance.
16. Importing Modules:
  - The code imports the `Sequential` class from the `keras.models` module and the `Dense` class from the `keras.layers` module.
  - It also imports the `MinMaxScaler` class from the `sklearn.preprocessing` module.
17. Feature Scaling:
  - A `MinMaxScaler` object named `scaler` is instantiated.
  - The `fit_transform()` method of the scaler is applied to the training data `X_train` to scale the features.
This method computes the minimum and maximum values of each feature and scales the data accordingly.
  - The `transform()` method of the scaler is applied to the test data `X_test` to scale the features. This ensures that the test data is scaled using the same scaling factors as the training data.
  - The scaled training data is stored in `X_train_scaled`, and the scaled test data is stored in `X_test_scaled`.
Feature scaling is a common preprocessing step in machine learning that helps to normalize the range of input features. The MinMaxScaler scales the features to a specified range, often between 0 and 1, based on the minimum and maximum values observed in the training data. This ensures that all features have a
similar scale and prevents some features from dominating others during model training.


### 2.3. Data Mining
Building upon the initial understanding of the data and its preprocessing, the next step is to apply data mining techniques to uncover meaningful insights and patterns. In this project, we leverage the power of machine learning and statistical techniques to explore relationships within our dataset and create predictive models.

1. **Regression Analysis**: Our project applies regression analysis to identify relationships between different variables in our dataset. Specifically, we focus on PM2.5, O3, and NO2, the three pollutants identified as the most impactful on human health and environmental quality. The regression analysis will provide us with a quantitative understanding of how these variables relate to each other and how they can influence future air quality predictions.

2. **Decision Trees and Random Forests**: These machine learning techniques are invaluable for identifying significant features and decision-making rules in the dataset. Our project leverages these tools to enhance the predictive power of our models and provide a clear, interpretable structure for understanding air quality determinants.

3. **Deep Learning Models**: Given the complex nature of environmental data and the multifaceted interactions between different air pollutants, we employ deep learning models. These models, characterized by their ability to learn high-level features from data, are well-suited for handling complex patterns and nonlinear relationships that might exist in our data.

### 2.4. Model Evaluation

Model evaluation is crucial to ensure that our predictive models are reliable and robust. This project utilizes several evaluation techniques to measure the performance of our predictive models.

1. **Cross-Validation**: To mitigate the risk of overfitting and to ensure our models generalize well to unseen data, we employ cross-validation techniques. This method divides the dataset into multiple subsets and tests the model on one subset while training it on the others. This process is repeated multiple times, providing a robust estimate of model performance.

2. **Error Metrics**: We will utilize Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and R^2 as primary error metrics. These metrics will help us quantify the performance of our models and understand their strengths and weaknesses.

3. **ROC Curves**: For classification problems, such as determining whether pollution levels will exceed a certain threshold, we will use Receiver Operating Characteristic (ROC) curves. This tool allows us to visualize the performance of our binary classifiers and helps to determine the optimal threshold.

4. **Confusion Matrix**: Another useful tool for classification problems is the confusion matrix, which provides a comprehensive overview of how our models perform on each class.

These evaluation techniques together will provide a comprehensive picture of our models' performance and help us refine and improve them further.
<br>
<br>


## 3.0 Data Mining

Data mining forms the crux of our research, providing us with the analytical methodologies necessary to extract valuable insights from our dataset. Following the path set by our understanding of the problem, dataset selection, and preprocessing, this section delves into the data mining process, techniques, tools, and challenges we encounter.

### 3.1. Data Mining Process

Data mining is an iterative process that allows us to derive meaningful insights from our dataset. To ensure the robustness and replicability of our research, we follow the Cross-Industry Standard Process for Data Mining (CRISP-DM). This systematic process commences with understanding the problem at hand, progressing through stages of data preparation, model creation, evaluation, and eventually culminating in the deployment of our predictive model.

### 3.2. Data Mining Techniques

Our dataset's complexity and diversity require a mix of data mining techniques to uncover hidden patterns and relationships effectively. Our project makes use of the following:

1. **Regression Analysis**: Leveraging the power of regression analysis, we establish relationships among variables, especially focusing on PM2.5, O3, and NO2, which are critical in forecasting air quality.

2. **Decision Trees and Random** Forests: These machine learning techniques assist us in identifying significant features and decision-making rules in the dataset, thus enriching the predictive capabilities of our models.

3. **Deep Learning Models**: Given the intricate nature of our dataset and the multi-dimensional interactions between air pollutants, we also employ deep learning models. These advanced models are proficient in discovering complex patterns and nonlinear relationships.

### 3.3. Data Mining Tools

Our research employs state-of-the-art tools to facilitate the data mining process. Python, a versatile programming language, forms the backbone of our project. We leverage libraries like pandas for data manipulation, NumPy for numerical computations, and Scikit-learn and TensorFlow for machine learning tasks.

### 3.4. Data Mining Challenges

Despite its power, data mining presents its own set of challenges. Issues like missing data, outliers, and noise can potentially interfere with our models' effectiveness. Overfitting, where a model performs excellently on training data but fails on unseen data, poses another challenge. We will apply best practices like cross-validation, regularization, and robust outlier detection methods to mitigate these issues.

### 3.5. Ethics in Data Mining

In line with our commitment to uphold the highest standards of integrity in our research, we ensure to adhere to ethical principles in data mining. This includes safeguarding the privacy and confidentiality of data, promoting non-discrimination, and maintaining transparency throughout the mining process.

In sum, the data mining segment of our research encompasses robust, ethical, and replicable processes aimed at deriving insightful revelations from our data. These insights are pivotal in constructing a reliable model for predicting future air quality in British Columbia, Canada, specifically focusing on PM2.5, O3, and NO2 levels.


<br>
<br>


## 4.0 Expected Findings
Our project is targeted towards the creation of a detailed predictive model capable of forecasting PM2.5, O3, and NO2 air quality levels in Victoria, British Columbia, Canada. This initiative relies heavily on the hourly air pollutant monitoring data and meteorological data, focusing on these three specific pollutants due to their significant implications on human health and the environment. Below are the expected findings from our study:

1. **Predictive Model**:

We aim to design a comprehensive model based on a decade-long dataset from 2011 to 2021. These datasets contain information from various air quality monitoring stations across British Columbia, ensuring that our forecasts are well-founded and inclusive of past trends. The successful development of this model will facilitate an understanding of the air quality dynamics, allowing for the prediction of future levels of these pollutants.

2. **Identification of Corelating Factors**:

Through our research, we expect to reveal significant correlations between these pollutants and variables such as meteorological conditions, industrial emissions, and urban development patterns. If, for instance, industrial emissions are strongly linked with high pollutant levels, this finding could guide future policies towards stricter emission controls on industries.

3. **Interactions between Pollutants**:

While focusing on PM2.5, O3, and NO2, our study will also examine the interactions between these pollutants. Understanding these interactions can provide insights into how changes in one pollutant level may affect the levels of others. This information could be invaluable in creating comprehensive air quality management strategies that consider the intricate relationships between different pollutants.

4. **Forecasting of Pollutant Levels**:

Based on identified correlations and interactions, our model will generate forecasts for future pollutant levels. This predictive capability will not only aid in anticipating potential air quality challenges but also in devising proactive measures to maintain or improve air quality.

5. **Basis for Comparative Analysis**:

Given the robust nature of our dataset, we believe that the insights obtained from our research will not only enrich our understanding of air quality dynamics in Victoria but also offer a valuable resource for comparison with other locations. Understanding how Victoria’s air quality dynamics compare with those of other regions can inform broader air quality management strategies and policies.

In conclusion, we expect our research to contribute significantly to understanding and managing air quality in Victoria, British Columbia, providing a comprehensive tool for forecasting air quality and a basis for comparison with other regions.


<br>
<br>


## 5.0 Implications and Future Work
Our research's potential impacts extend beyond the development of a predictive model for air quality in Victoria, British Columbia. The findings can provide significant insights to various stakeholders, including policymakers, environmentalists, and scientists, and establish a basis for future research. Here are the potential implications and areas for future work:

### 5.1. Policy Recommendations

Our study's insights will be a valuable resource for policymakers who need to make informed decisions about air quality management. Understanding the correlations between various factors like industrial emissions, meteorological conditions, and pollutant levels can guide the development of more targeted and effective policies. For instance, if our findings highlight a strong relationship between industrial emissions and high pollutant levels, stricter regulations on industry emissions could be a viable policy direction.

### 5.2. Public Health Implications

Public health officials can use our research findings to better understand the potential risks associated with poor air quality, focusing on PM2.5, O3, and NO2, and devise strategies to mitigate these risks. This could involve implementing public awareness campaigns about the health risks associated with these pollutants or advising individuals with respiratory issues to limit outdoor activities during periods of poor air quality.

### 5.3. Future Research Directions

The methodology and findings from our study can guide future research in this field. Researchers looking to extend our work could consider incorporating additional factors that could influence air quality, such as traffic data or local population density. Another potential direction could be to explore the applicability of our model to other regions, adjusting for local variables as necessary.

### 5.4. Technological Enhancements

The model we develop could serve as a foundation for the development of advanced air quality monitoring and prediction tools. For example, real-time monitoring systems could integrate our model to provide more accurate forecasts, enabling rapid response to changes in air quality.

### 5.5 Climate Change Mitigation

Finally, our research may contribute to broader climate change mitigation efforts. Understanding the key drivers of air pollutant levels can help in designing strategies to reduce greenhouse gas emissions and combat climate change.

In summary, our research has wide-ranging implications for policy-making, public health, future research, technological advancements, and climate change mitigation. We anticipate our work providing a strong basis for continued exploration and progress in these areas.


<br>
<br>


## 6.0 Conclusion
The importance of maintaining good air quality and its impact on public health and the environment cannot be overstated. Recognizing this, we embarked on a comprehensive research project aimed at developing a predictive model for forecasting air quality in Victoria, British Columbia, with a specific emphasis on three pollutants - PM2.5, O3, and NO2. These pollutants, owing to their severe health and environmental implications, were the primary focus of our study.

After meticulous data gathering, preprocessing, and rigorous exploration of machine learning techniques, we anticipate our research will yield a model capable of providing accurate forecasts of air quality. Our model, based on linear regression, logistic regression, decision tree, and gradient boosting, aims to predict the future levels of PM2.5, O3, and NO2, utilizing historical and current data and taking into consideration a multitude of factors. We predict the successful detection of trends and relationships between these factors and the levels of pollutants in the atmosphere.

The findings from our study have far-reaching implications. For policymakers, the insights derived from our study will act as a tool to guide more effective and targeted air quality management strategies. For public health officials, the study can offer insights into periods of potentially harmful air quality, enabling them to advise and protect those at greatest risk.

Looking beyond the immediate application of our model, we see vast potential for future research. Whether it's extending our model to include additional variables or exploring its applicability to other geographical locations, our research paves the way for a deeper exploration into the field of air quality prediction. The development of real-time monitoring systems that integrate our model could also be a game-changing advancement in technology.

Moreover, our research potentially contributes to the global effort to combat climate change by aiding in the identification of key pollution drivers. This knowledge can guide interventions designed to reduce harmful emissions and mitigate the impacts of climate change.

In conlcusion, our research is not merely an academic endeavor but has wide-ranging implications for the wellbeing of our society and the environment. The potential of our predictive model extends beyond its immediate application and lays the groundwork for future research, technological advancements, and proactive interventions to safeguard our environment and public health. This project thus stands as a testament to the significance of data mining and predictive modeling in addressing pressing environmental challenges.

We remain committed to adhering to ethical data mining principles and transparent procedures throughout this process, ensuring the highest levels of accuracy, reliability, and credibility of our work. As we continue refining our model and uncovering deeper insights, we are confident that our work will make a significant contribution to the discourse on air quality management and environmental conservation in Victoria, British Columbia, and beyond.

<br>
<br>


## 7.0 Links
Data Source:
https://catalogue.data.gov.bc.ca/dataset/air-quality-monitoring-verified-hourly-data
