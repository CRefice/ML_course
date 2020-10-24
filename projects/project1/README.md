# Running the prediction model

We provide the script `run.py` which has been used to generate the CSV file submitted to AIcrowd.

The script takes three command-line arguments, all of which are paths:
1. The training set CSV file
2. The test set CSV file
3. The file in which to write predictions for the test set

The files are to be provided in the above order. An example run of the script would thus look like the following:
```bash
python run.py data/train.csv data/test.csv data/predictions.csv
```

Note that the scripts uses the modules `implementations.py` and `proj1_helpers.py`, both of which have been added to this zip file. It's important for all three `.py` files to be in the same directory, or the code will not run.

This code has been tested on the latest version of python (3.8.6 at the time of writing), but it should be able to run on any 3.x python version.

## Outline of prediction process

More information can be found in the comments of run.py, but the outline of the prediction process is as follows:

1. Read the training data.
2. Partition it into three groups based on the values of PRI_jet_num. [0, 1, 2&3]
3. For each group, normalize the training data's columns and record their mean and standard deviation. Further augment the transformed data using degree-10 polynomial expansion.
4. Train each group as a separate model using ridge regression.
5. Read the test data.
6. Partition the test data analogously to step 2.
7. Transform the test data analogously to point 3, using the same mean and standard deviation as computed from the training data to ensure equal transformations.
8. Apply the weights obtained in step 4 to the data obtained in step 7.
9. Clamp the obtained prediction to the label set {-1, 1}.
10. Save the predictions to file.

# Validation
As mentioned in the report, we performed model selection through nested cross-validation.
Even though we did not end up using logistic regression in our final model, we had to compare it to other methods to come to that conclusion. Since the logistic regression method implemented in the labs only works with labels in the set {0, 1}, we had to transform the training labels from the set {-1, 1} to match the expected form, using the formula `transformed = (original + 1) / 2`. Luckily, other classification methods could use any feature set, so we only had to transform the labels once.
