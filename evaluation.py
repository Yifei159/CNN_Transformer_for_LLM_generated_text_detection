import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, accuracy_score

def evaluate_and_display_model_performance(y_true, y_pred_probs, dataset="Test"):
    y_pred = (y_pred_probs > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = tp / (tp + ((fn + fp) / 2))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Human", "LLM"], cmap=plt.cm.Blues)
    disp.ax_.set_title(f"Confusion Matrix on {dataset} Dataset -- F1 Score: {round(f1, 2)}")
    plt.show()
    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_probs)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc="lower right")
    plt.show()