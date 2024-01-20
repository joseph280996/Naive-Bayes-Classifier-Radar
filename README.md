# Naive Bayes Classification
## Prerequisite
```
python 3.12.0
```

## Assumptions
- The training file will only consists of 20 data points, 1st half of it is birds measurements and the other half is for planes measurements.
- Each entry in the testing file must be a list of velocities where the sample rate is every 1s for a total of 600 seconds.
- User can specify the whether to enable the extra feature or not with better accuracy and prediction.

## Analysis

- Looking at the training.txt file, the majority of the birds velocity mean falls in the 40-70 range.
- However, there are instances where the bird (probably a different species) has velocity up to 80-100 range (Bird 3 and 4) which is outside an irregular in the bird training data.
- If the current categorized method was kept, it will pull up the mean and skewed the probability.
- Re-categorized of the birds was made into these categories:
  - Bird 3 and 4 as Bird2
  - The rest are Bird1

- Another observation was that the plane has multiple range that was recorded:
  - Mean velocity < 70 (Plane 1 + 5)
  - 70 <= mean velocity < 85 (Plane 2 + 6)
  - 85 <= mean velocity < 110 (Plane 4 and 10)
  - Mean velocity >= 110 (the rest)
- With that, these ranges are different type of planes and can be categorized with Plane 1 - 4 in the order list above

- Also, the probability graph of birds and plane given its velocity is closely to the normal distribution. Therefore, the probability density function is going to use the Guassian Distribution formula to calculate the likelihood given a velocity.

## Example comamnds to run

Normal run using given likelihood against test file:

```
python3 main.py
```

Normal run using given likelihood against training file:

```
python3 main.py -pat
```

Enable training and predicting testing file:

```
python3 main.py -eaf
```

Enable training and predicting training file:

```
python3 main.py -eaf -pat
```

Run with custom training file:

```
python3 main.py -t data_files/training
```

Run with custom testing file:

```
python3 main.py -t data_files/test
```
