# nlptd3

TD 3 - NLP Class
Sept 2020

## Dataset

To load both train and test datasets in your python script, you may use the python library Pandas:

```python
import pandas

train_set = pd.read_csv("train_dataset.csv")
test_set = pd.read_csv("test_dataset.csv")
```

This will give you two `pandas.DataFrame` with a label and a features columns.

**Example:**

|label|features|
|-|-|
|'pos'|'a positive movie review...'|
|'neg'|'a negative movie review...'|
|...|...|
