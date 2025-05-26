import torch 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for activations, labels in dataloader:
            activations, labels = activations.to(device), labels.to(device)
            outputs = model(activations)
            predictions = (outputs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
