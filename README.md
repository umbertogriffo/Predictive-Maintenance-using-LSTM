# Recurrent Neural Networks for Predictive Maintenance
* Author: Umberto Griffo
* Twitter: @UmbertoGriffo

## Colab
You can try the code directly on [Colab](https://colab.research.google.com/drive/1tjIOud2Cc6smmvZsbl-QDBA6TLA2iEtd).
Save a copy in your drive and enjoy It!

## Software Environment

### Updated Requirements (2025-10-08)
* Python >= 3.11
* TensorFlow >= 2.15.0
* NumPy >= 1.24.0
* pandas >= 2.0.0
* scikit-learn >= 1.3.0
* matplotlib >= 3.7.0
* h5py >= 3.9.0
* Pillow >= 10.0.0

**Installation:**
```bash
pip install -r requirements.txt
```

<details>
<summary>Original Environment (Legacy)</summary>

* Python 3.6
* numpy 1.13.3
* scipy 0.19.1
* matplotlib 2.0.2
* spyder 3.2.3
* scikit-learn 0.19.0
* h5py 2.7.0 
* Pillow 4.2.1 
* pandas 0.20.3
* TensorFlow 1.3.0
* [Keras 2.1.1](https://keras.io)

**Note:** Original environment is outdated and no longer recommended.
</details>

## Problem Description
In this example, I build an LSTM network in order to predict remaining useful life (or time to failure) of aircraft engines <a href="https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan">[3]</a> based on the scenario described at <a href="https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb">[1]</a> and <a href="https://gallery.azure.ai/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2">[2]</a>.
The network uses simulated aircraft sensor values to predict when an aircraft engine will fail in the future allowing maintenance to be planned in advance.
The question to ask is "Given these aircraft engine operation and failure events history, can we predict when an in-service engine will fail?"
We re-formulate this question into two closely relevant questions and answer them using two different types of machine learning models:

	* Regression models: How many more cycles an in-service engine will last before it fails?
	* Binary classification: Is this engine going to fail within w1 cycles?

## Data Summary
In the **Dataset** directory, there are the training, test and ground truth datasets.
The training data consists of **multiple multivariate time series** with "cycle" as the time unit, together with 21 sensor readings for each cycle.
Each time series can be assumed to be generated from a different engine of the same type.
The testing data has the same data schema as the training data.
The only difference is that the data does not indicate when the failure occurs.
Finally, the ground truth data provides the number of remaining working cycles for the engines in the testing data.
The following picture shows a sample of the data: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/datasetSample.png"/>
</p>
You can find more details about the data at <a href="https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb">[1]</a> and <a href="https://gallery.azure.ai/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2">[2]</a>.

## Experimental Results
### Results of Regression model

|Mean Absolute Error|Coefficient of Determination (R^2)|
|----|----|
|12|0.7965|

The following pictures show the trend of loss Function, Mean Absolute Error, R^2 and actual data compared to predicted data: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_regression_loss.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_mae.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_r2.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_regression_verify.png"/>
</p>
         
### Results of Binary classification 

|Accuracy|Precision|Recall|F-Score|
|----|----|----|----|
|0.97|0.92|1.0|0.96|

The following pictures show the trend of loss Function, Accuracy and actual data compared to predicted data: 
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_loss.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_accuracy.png"/>
</p>
<p align="center">
  <img src="https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/blob/master/Output/model_verify.png?raw=true"/>
</p>

## Extensions
We can also create a model to determine if the failure will occur in different time windows, for example, fails in the window (1,w0) or fails in the window (w0+1, w1) days, and so on. This will then be a multi-classification problem, and data will need to be preprocessed accordingly. 

## Who is citing this work?

* In chapter 10 of [Hands-On Artificial Intelligence for IoT](https://www.amazon.it/Hands-Artificial-Intelligence-IoT-techniques/dp/1788836065) book
	* https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-IoT/tree/master/Chapter10
* In chapter 7 of [Mobile and Wireless Communications with Practical Use-Case Scenarios](https://www.amazon.com/Wireless-Communications-Practical-Use-Case-Scenarios/dp/1032119020) book
	* https://www.google.pt/books/edition/Mobile_and_Wireless_Communications_with/lvqhEAAAQBAJ?hl=en&gbpv=1
* In `Using Recurrent Neural Networks to predict the time for an event` master's thesis (Universitat de Barcelona, Barcelona, Spain). Retrieved from [here](http://diposit.ub.edu/dspace/bitstream/2445/134691/3/memoria.pdf)
* In `Exploring Cloud Assisted Tiny Machine Learning Application Patterns for PHM Scenario`. Retrieved from [here](https://www.dre.vanderbilt.edu/~gokhale/WWW/papers/PHM21_TinyML_Prognostics.pdf) and [here](https://www.researchgate.net/publication/356519569_The_Future_of_PHM_Could_be_Tiny_under_Cloud_Exploring_Potential_Application_Patterns_of_TinyML_in_PHM_Scenarios)

## Recent Updates (2025-10-08)

### ✅ Critical Fixes & Improvements

This project has been audited and modernized with the following improvements:

**🔧 Bug Fixes:**
- ✅ Fixed Issue #8: Keras metric naming compatibility (`'acc'` → `'accuracy'`)
- ✅ Replaced deprecated `predict_classes()` method (TensorFlow 2.6+ compatible)
- ✅ Fixed reproducibility issues with proper seed management
- ✅ Corrected typo: 'Accurracy' → 'Accuracy'

**🏗️ Infrastructure:**
- ✅ Modern dependency management (`requirements.txt`)
- ✅ Configuration system (`configs/config.yaml`)
- ✅ Modular utilities (`src/utils/`)
- ✅ Error handling framework
- ✅ Proper `.gitignore` for version control

**📚 Documentation:**
- ✅ Comprehensive `CHANGELOG.md`
- ✅ Project context in `memory.json`
- ✅ Detailed `AUDIT_REPORT.md`
- ✅ Validation script (`validate_fixes.py`)

**📊 Compatibility:**
- ✅ TensorFlow 2.15+ / Keras 3.x compatible
- ✅ Python 3.11+ support
- ✅ Backward compatible with older Keras versions
- ✅ Modern ML engineering practices

### Project Structure

```
Predictive-Maintenance-using-LSTM/
├── configs/
│   └── config.yaml              # Configuration management
├── src/
│   ├── lstm/
│   │   ├── binary_classification.py  # Fixed & updated
│   │   └── regression.py
│   └── utils/                   # Shared utilities
│       ├── data_loader.py       # Data loading with error handling
│       ├── preprocessor.py      # Preprocessing pipeline
│       └── reproducibility.py   # Seed management
├── Dataset/                     # NASA C-MAPSS data
├── Output/                      # Models & visualizations
├── requirements.txt             # Modern dependencies
├── .gitignore                   # Version control
├── CHANGELOG.md                 # Change tracking
├── AUDIT_REPORT.md              # Comprehensive audit results
└── validate_fixes.py            # Validation script
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM.git
cd Predictive-Maintenance-using-LSTM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Validate installation
python validate_fixes.py

# Run binary classification
python src/lstm/binary_classification.py

# Run regression model
python src/lstm/regression.py
```

### Configuration

All hyperparameters and paths are now centralized in `configs/config.yaml`:

```yaml
data:
  preprocessing:
    w0: 15                    # Early warning window
    w1: 30                    # Critical failure window
    sequence_length: 50       # LSTM time steps

model:
  classification:
    lstm_units: [100, 50]
    dropout_rate: 0.2
    optimizer: adam
```

### Reproducibility

For reproducible results:

```python
from utils.reproducibility import set_random_seeds

# Set all random seeds
set_random_seeds(1234)

# For complete reproducibility, also set environment variable:
# export PYTHONHASHSEED=0  (Linux/Mac)
# set PYTHONHASHSEED=0     (Windows)
```

## References

- [1] Deep Learning for Predictive Maintenance https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
- [2] Predictive Maintenance: Step 2A of 3, train and evaluate regression models https://gallery.azure.ai/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2
- [3] A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan), NASA Ames Research Center, Moffett Field, CA 
- [4] Understanding LSTM Networks http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Contributing

Contributions are welcome! Please see:
- `CHANGELOG.md` for recent changes
- `AUDIT_REPORT.md` for improvement details
- `CONTRIBUTION_PLAN.md` for future roadmap

## License

See `LICENSE` file for details.
