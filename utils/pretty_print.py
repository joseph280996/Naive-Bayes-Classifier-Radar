def pretty_print(predictions: list[str]):
    """
    Pretty printing the prediction list

    Arguments:
        predictions: the list of predictions
    """
    for i, result in enumerate(predictions):
        if result == "Bird1" or result == "Bird2":
            print(f"Object {i + 1}: Bird")
        else:
            print(f"Object {i + 1}: Plane")
