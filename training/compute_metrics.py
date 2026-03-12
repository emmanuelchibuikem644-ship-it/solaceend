
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    # Weighted precision/recall/F1 (optional, you can report separately)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    # Macro precision/recall/F1
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )

    # Micro precision/recall/F1
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="micro"
    )

    # Accuracy
    accuracy = accuracy_score(labels, predictions)

    # Hamming loss (good for multi-label datasets)
    ham_loss = hamming_loss(labels, predictions)

    return {
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "hamming_loss": ham_loss
    }