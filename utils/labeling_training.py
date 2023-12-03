from statistics import mean


def labeling_training_data_for_extra_feature(
    training_data: list[list[float]], split_ratio: float
):
    """
    This function handling labeling the training data according to analysis described in the README
    It will take in split ratio to determine how much is for bird and how much is for plane.
    Example:
        split_ratio = 0.5 meaning first half Birds and other half Plane

    Arguments:
        training_data: list of the content of the training file
        split_ratio: the ratio amount of Bird and Plane of the training file (default: 0.5)

    Returns:
        The list of label in order with the order of each training feature
    """
    split_idx = int(len(training_data) * split_ratio)
    labels = []

    # Labeling Birds
    for obj in training_data[:split_idx]:
        mean_val = mean(obj)
        if mean_val > 80:
            labels.append("Bird2")
        else:
            labels.append("Bird1")

    # Labeling Planes
    for obj in training_data[split_idx:]:
        mean_val = mean(obj)
        if mean_val < 70:
            labels.append("Plane1")
        elif mean_val < 85:
            labels.append("Plane2")
        elif mean_val < 110:
            labels.append("Plane3")
        else:
            labels.append("Plane4")

    return labels
