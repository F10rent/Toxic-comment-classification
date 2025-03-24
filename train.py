import torch
from tqdm import tqdm
from collections import Counter
import warnings
import numpy as np

def find_longest_length(tokens):
    """
    Find the longest tokenized text in the dataset.

    :param tokens: A pandas Series where each element is a list of tokens.
    :returns: The length of the longest tokenized text.
    """
    return max(tokens.apply(len))


def find_avg_sentence_length(tokens):
    """
    Compute the average number of tokens per text.

    :param tokens: A pandas Series where each element is a list of tokens.
    :returns: The average tokenized sentence length.
    """
    return tokens.apply(len).mean()


from collections import Counter

def find_word_frequency(tokens_lists, most_common=None):
    """
    Calculate the frequency of words in a dataset.

    :param tokens_lists: A list of lists, where each inner list contains tokenized words.
    :param most_common: Number of top words to return. If None, return all.
    :returns: A list of tuples (word, frequency), sorted by frequency.
    """
    # Flatten the list of tokens into a single list
    corpus = [word for token_list in tokens_lists for word in token_list] 
    # Count word frequency
    count_words = Counter(corpus)

    # Get most common words
    word_frequency = count_words.most_common(n=most_common) 
    return word_frequency



def filter_low_freq_words(words_freq, min_freq):
    """
    Counts and returns a list of high frequency words in (word, count) format.
    """
    filtered_words = []
    # 只保留频率 ≥ min_freq 的词
    for word, count in words_freq:
      if count >= min_freq:
        filtered_words.append((word, count))
      else:
        break

    return filtered_words


# def get_max_len(texts, tokenizer):
#     # gaigaigai
#     tokens_list = tokenizer.tokenize(texts)
#     token_lengths = [len(tokens) for tokens in tokens_list]  
#     max_len = int(np.percentile(token_lengths, 97))  
#     return max_len

def word2int(input_words, num_words):
    """
    Create a dictionary mapping words to integers.
    
    :param input_words: List of tuples [(word, frequency)].
    :param num_words: Number of top words to keep.
    :return: Dictionary {word: index}.
    """
    if num_words > -1:
        int_mapping = {w: i+1 for i, (w, _) in enumerate(input_words) if i < num_words}
    else:
        int_mapping = {w: i+1 for i, (w, _) in enumerate(input_words)}
    return int_mapping

def count_correct_incorrect(labels, outputs, running_correct):
    predictions = (outputs >= 0.5).int()
    labels = labels.int()

    for i in range(labels.size(1)):
        running_correct[i] += (predictions[:, i] == labels[:, i]).sum().item()

    return running_correct

def get_device():
    """Check if a GPU is available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function.
def train_model(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    
    # Initialize the correct prediction statistics for each label
    num_labels = trainloader.dataset[0]['label'].shape[0]  # Get the number of tags
    train_running_correct = [0] * num_labels

    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        inputs, labels = data['input'], data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        
        train_running_correct = count_correct_incorrect(labels, outputs, train_running_correct)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
    
    total_samples = len(trainloader.dataset)
    label_accuracies = [100. * (correct / total_samples) for correct in train_running_correct]
    
    epoch_loss = train_running_loss / counter
    
    return epoch_loss, label_accuracies



# Validation function.
def validate(model, validloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    
    num_labels = validloader.dataset[0]['label'].shape[0]  
    valid_running_correct = [0] * num_labels

    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(validloader), total=len(validloader)):
            counter += 1
            inputs, labels = data['input'], data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            
            valid_running_correct = count_correct_incorrect(labels, outputs, valid_running_correct)
        
    total_samples = len(validloader.dataset)
    label_accuracies = [100. * (correct / total_samples) for correct in valid_running_correct]
    
    epoch_loss = valid_running_loss / counter
    
    return epoch_loss, label_accuracies


# if __name__ == '__main__':
#     warnings.filterwarnings("ignore")
#     config = get_config()
#     train_model(config)