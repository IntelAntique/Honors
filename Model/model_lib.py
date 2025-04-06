import torch
from torchmetrics import ROC
import matplotlib.pyplot as plt
from torchmetrics.functional import auroc

def roc(dataloader, model, device, n):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in dataloader:
            X, y = data[:, : n].type(torch.float).to(device), data[:, n].type(torch.long).to(device)
            pred = model(X)
            all_preds.append(pred[:, 1])
            all_targets.append(y)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    roc = ROC(task="binary")
    fpr, tpr, thresholds = roc(all_preds, all_targets)
    auc = auroc(all_preds, all_targets, task="binary")

    print(f"FPR: {fpr}")
    print(f"TPR: {tpr}")
    print(f"Thresholds: {thresholds}")

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc.item())
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()