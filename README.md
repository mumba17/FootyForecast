# FootyForecast
FootyForecast is a project that aims to create a football match prediction model. The datasets used in this project are scraped from FBref. Currently a work in progress, mainly focussing on the Neural Network branche in the pytorch folder. 

# TODO
Currently I am working on updating the data with more datasets from FBref, trying to improve the models accuracy through that. I am adding shooting/defending statistics which the dataset currently does not use. 

I must add that the model is not very user-friendly and I will attempt to improve this in the near future.

# Getting Started
To get started with FootyForecast, you will need to clone the repository to your local machine. You can do this by running the following command:

```git clone https://github.com/mumba17/FootyForecast/```

Once you have cloned the repository, you can navigate to the project directory and install any required dependencies.

# Usage
To use FootyForecast, you will need to provide the python script with your data in the ```pytorch_neural_test_model.py``` file. If the correct data format is not provided this model will NOT work. I have created a ```generate_dataset.py``` script that should convert the correct FBref tables to useable data. Please refer to the ```datasets``` folder for correct implementation.

# Contributing
If you would like to contribute to FootyForecast, please feel free to submit a pull request or open an issue on the GitHub repository.
