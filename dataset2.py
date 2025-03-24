import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset, Dataset

def split_train_valid(train_data_path, valid_split=0.1, seed=42):
    df = pd.read_csv(train_data_path)  
    dataset_size = len(df)
    
    valid_size = int(valid_split * dataset_size) 

    torch.manual_seed(seed)
    indices = torch.randperm(len(df)).tolist() 

    df_train = df.iloc[indices[:-valid_size]].reset_index(drop=True)  
    df_valid = df.iloc[indices[-valid_size:]].reset_index(drop=True)

    return df_train, df_valid

def get_datasets(train_data, valid_data, test_data, tokenizer, max_len=350):
    dataset_train = CommentClassificationDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_len=max_len
    )
    dataset_valid = CommentClassificationDataset(
        data=valid_data,
        tokenizer=tokenizer,
        max_len=max_len
    )
    dataset_test = CommentClassificationDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_len=max_len,
        is_test=True
    )
    print(f"Number of training samples: {len(dataset_train)}")
    print(f"Number of validation samples: {len(dataset_valid)}")
    print(f"Number of test samples: {len(dataset_test)}")

    return dataset_train, dataset_valid, dataset_test


def get_dataloaders(dataset_train, dataset_valid, dataset_test, BATCH_SIZE=8):
    train_loader = DataLoader(
        dataset_train, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=4
    )
    valid_loader = DataLoader(
        dataset_valid, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=4
    )
    test_loader = DataLoader(
        dataset_test, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    return train_loader, valid_loader, test_loader


class CommentClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=350, is_test=False):
        """
        :param data: The CSV data, containing the 'comment_text' and label columns  
        :param tokenizer: The tokenizer function to be used  
        :param vocab: Pre-built vocabulary word2int  
        :param max_len: The predefined maximum text length   

        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_column = ['comment_text']
        self.label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.is_test = is_test
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.loc[idx, self.text_column])
        input = self.tokenizer.encode(text)

        # Fill or truncate each sublist to max_len
        if len(input) < self.max_len:
            # padding
            input_tensor = torch.cat(
                [
                    torch.tensor(input, dtype=torch.int32),
                    torch.tensor([self.pad_token] * (self.max_len - len(input)), dtype=torch.int32),
                ],
                dim = 0,
            )
        else:
            input_tensor = torch.tensor(input[:self.max_len], dtype=torch.int32)

        if self.is_test:
            return {"input": input_tensor}
        
        label = self.data.iloc[idx][self.label_columns].values.astype(float)
        label = torch.tensor(label, dtype=torch.float32)

        return {
            "input":  input_tensor, 
            "label": label,
        }    
