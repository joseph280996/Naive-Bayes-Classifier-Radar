# Naive Bayes Classification

## Assumptions

- The python version that is being used for testing is 3.12+
- The training file will only consists of 20 data points, 1st half of it is birds measurements and the other half is for planes measurements.
- Each entry in the testing file must be a list of velocities where the sample rate is every 1s for a total of 600 seconds.
- User can specify the whether to enable the extra feature or not with better accuracy and prediction.

## Analysis

- Looking at the training.txt file, we can notice that the majority of the birds velocity mean falls in the 40-70 range.
- However, there are instances where the bird (probably a different species) has velocity up to 80-100 range (Bird 3 and 4) which is outside an irregular in our bird training data.
- If we keep this the same as regular bird category then it will pull up the mean and skewed our probability.
- I've decided to re-categorized the birds like so:

  - Bird 3 and 4 as Bird2
  - The rest are Bird1

- Another observation that I saw was that the plane has multiple range that was recorded:
  - mean velocity < 70 (Plane 1 + 5)
  - 70 <= mean velocity < 85 (Plane 2 + 6)
  - 85 <= mean velocity < 110 (Plane 4 and 10)
  - mean velocity >= 110 (the rest)
- With that, I figure that all of these are different type of planes and categorize them with Plane 1 - 4 in the order list above

- Also, I notice that the probability graph of birds and plane given its velocity is closely to the normal distribution. Therefore, the probability density function is going to use the Guassian Distribution formula to calculate the likelihood given a velocity.

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
