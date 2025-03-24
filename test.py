import torch
from tqdm import tqdm

def model_predict(model, testloader, device):
    model.eval()  
    print('Testing')
    
    all_predictions = []
    with torch.no_grad():  
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            inputs = data['input']
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
         
            predictions = (outputs >= 0.5).int()
            all_predictions.append(predictions.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    return all_predictions


def cul_acc(labels, predctions):
    labels = labels.values.tolist()
    num_labels = len(labels[0]) - 1
    correct_labels = [0] * num_labels
    total_labels = 0
    for x, y in zip(labels, predctions):
        if x[1] == -1:
            continue
        else:
            total_labels += 1
            for i in range(num_labels):
                if x[i+1] == y[i]:
                    correct_labels[i] += 1
    
    return [t / total_labels for t in correct_labels]
    