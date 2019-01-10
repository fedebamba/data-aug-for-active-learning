def decide_confidence(vector, num_of_classes=10, norm=False):
    """
    :param vector: 2D vector - containing the network guesses for each image
    :param num_of_classes: int - the number of classes in the dataset; default 10
    :param norm: bool - if true the confidence will be normalized to 1; default False
    :return: 1D vector - for each image, the number of concording guesses; if norm is True, this number will be normalized
    """
    max_l = [max([(len([i for i in el if i == j])) for j in range(num_of_classes)]) for el in vector]
    return max_l if not norm else [float(el/len(vector[0])) for el in max_l]



