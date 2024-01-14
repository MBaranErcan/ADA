using LinearAlgebra, Zygote, ForwardDiff, Printf
using CSV, DataFrames
using StatsBase: mean
using Parameters
using Distributions
using Random
using Flux
using MLUtils
using NNlib  
using Random
using PyCall 

## Aim of this project is to predict whether a person has heart disease or not.
## We will use a neural network to predict the heart disease.
## We will use the heart-disease.csv file to train our network.
## We will use Flux.jl to create our neural network.
## We will use ADAM as the optimizer.

#### ----- ###
#### Before getting started you should write your student_number in integer format
 student_number::Int64 = 2881055520  ## <---replace 0 by your student_number 
### ---- ###

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


cd(@__DIR__) ##Change the dir to your current dir!!

data = CSV.read("heart-disease.csv", DataFrame) ## Our data has 12 columns and 918 rows.


# Shuffle the dataset
data = data[shuffle(1:end), :]

# Define features and target
data_x = convert(Array, Matrix(data[:, 1:11])) # features
data_y = convert(Array, data[:, 12]) # target

data_x = (data_x .- mean(data_x, dims = 2))./std(data_x, dims = 2)
data_y = (data_y .- mean(data_y))./std(data_y)

# Define the neural network
Network = let 
    Random.seed!(0) ## Set the seed for reproducibility.
    Chain(## Start with 11 inputs because we have 11 features (12 - result column).
        Dense(11, 8, σ),
        Dense(8, 1, σ),
        σ)
end

## Optimizer state --> we used ADAM as the optimizer
opt_state = Flux.setup(Flux.Adam(0.01), Network) 

## Shuffle the data
indexes = collect(1:size(data_x, 1)) ## data_x is a matrix so we use size(data_x, 1) to get the row count.
let
    Random.seed!(0) ## Set the seed for reproducibility.
    shuffle!(indexes) ## Shuffle the indexes.
end

# Split the data into train and test
train_indexes, val_indexes = indexes[101:end], indexes[1:100]
X_train, X_val, y_train, y_val = data_x[train_indexes, :], data_x[val_indexes, :], data_y[train_indexes], data_y[val_indexes]


train_data = Flux.Data.DataLoader((permutedims(X_train), y_train), batchsize = 16) ## Create a dataloader for the training data
val_data = Flux.Data.DataLoader((permutedims(X_val), y_val), batchsize = 8) ## Create a dataloader for the validation data


function train_network()
    elapsed_time = @elapsed begin   ## Start the timer.
        for epoch in 1:1000         ## Train the network for 1000 epochs.
            temp_val::Float32 = 0.f0
            temp_size::Int32 = 0
            trainmode!(Network)         ## Dropout layers active when in train mode
            for (x,y) in train_data     ## Grab the data from dataloader
                val, grads = Zygote.withgradient(Network) do Network    ## Take the gradient here.
                    mean(Network(x) - transpose(y) .|> abs)
                end 
                temp_val += val        ## Add the loss to the temp_val.
                temp_size += size(x)[end]  ## Add the size of the batch to the temp_size.
                Flux.update!(opt_state, Network, grads[1]) ##update the weights
            end
            temp_validation_val::Float32 = 0.f0 ## Create a temp variable for validation loss.
            temp_validation_size::Int32 = 0   ## Create a temp variable for validation size.
            testmode!(Network) ## Dropout layers inactive when in test mode.
            for (x,y) in val_data ## Grab the data from dataloader.
                temp_validation_val += mean(Network(x) - transpose(y)).^2   ## Calculate the validation loss.
                temp_validation_size += size(x)[end] ## Add the size of the batch to the temp_validation_size.
            end
            if epoch % 20 == 0  ## Print the loss every 20 epochs.
                println("Epoch $epoch: Train loss = $(temp_val/temp_size), Validation loss = $(temp_validation_val/temp_validation_size)") 
            end
        end
    end
    println("The training took $elapsed_time seconds.") ## Print the elapsed time.
end




train_network() ## Train the network.







# No need to run below.
if abspath(PROGRAM_FILE) == @__FILE__
    @assert student_number != 0
    println("Seems everything is ok!!!")
end