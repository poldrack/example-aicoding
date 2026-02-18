from sklearn.metrics import f1_score, precision_score, recall_score



def calculate_f1_score(y_true, y_pred):

    """

    Calculate the F1-score for a binary classification problem

    Args:

    y_true: list of true labels (binary values)

    y_pred: list of predicted labels (binary values)



    Returns:

    f1: float, F1-score

    """

    f1 = f1_score(y_true, y_pred)

    return f1



def main():

    # Example true and predicted labels for a binary classification problem

    y_true = [0, 0, 1, 1, 1, 1, 0, 1, 1, 0]

    y_pred = [0, 1, 1, 1, 0, 1, 0, 1, 0, 0]



    f1 = calculate_f1_score(y_true, y_pred)



    print("F1-score: {:.2f}".format(f1))



if __name__ == "__main__":

    main()
