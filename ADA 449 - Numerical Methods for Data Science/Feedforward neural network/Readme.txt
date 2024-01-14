## Aim of this project is to predict whether a person has heart disease or not.
## We will use a neural network to predict the heart disease.
## We will use the heart-disease.csv file to train our network.
## We will use Flux.jl to create our neural network.
## We will use ADAM as the optimizer.


### The CSV file includes the following 12 columns and 918 rows:
### 1. age
### 2. Sex (0 = Male, 1 = Female)
### 3. ChestPainType Represents the type of chest pain experienced by the individual. (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic) , angina means chest pain.
### 4. RestingBP: Represents the resting blood pressure of the individual. (in mmHg)
### 5. Cholesterol: Represents the cholesterol level of the individual. (in mg/dl)
### 6. FastingBS: Represents the fasting blood sugar level of the individual. (1 = if FastingBS > 120 mg/dl, 0 = otherwise)
### 7. RestingECG: Represents the resting electrocardiographic results of the individual. (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy)
### 8. MaxHR: Represents the maximum heart rate achieved by the individual during exercise. (in bpm)
### 9. ExerciseAngina: Represents whether the individual experienced exercise-induced angina (chest pain) or not. (1 = yes, 0 = no) (An)
### 10.Oldpeak: Simply means the previous peak. It is the difference between the lowest value to which the heart rate falls after exercise and the peak heart rate during exercise. The bigger the value is, the more abnormal the heart rate of the individual is considered to be.
### 11.ST_Slope: Represents the slope of the peak exercise ST segment. (0 = going up, 1 = flat, 2 = going down)
### 12.HeartDisease: Represents the presence or absence of heart disease in the individual. (1 = yes, 0 = no)
### I acquired this csv dataset from a github user named Syed-Owais-Noor, but i rearranged columns from string values to labeled integer values.
### My program first shuffles the data, then normalizes the data, then splits the data into train and test sets, then trains the network, then tests the network.