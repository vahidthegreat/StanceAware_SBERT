import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, losses
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType # peft-0.7.1
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    AutoConfig,
)

removal_cutoff_contrastive = 0.5
removal_cutoff_triplet = 0.3
# removal_cutoff = 0.0

from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q", "k","v","o"],
        bias="none",
        lora_dropout=0.05, # Conventional
#         task_type="CAUSAL_LM",
#         modules_to_save = ["lm_head", "embed_tokens"]   # because we added new tokens
        )



meta_data = pd.read_excel('Datasets/Kialo_MetaData_all.xlsx')
meta_data = meta_data[meta_data['language']=='en']
post_id_set = list(set(list(meta_data["post_id"])))

# Initialize empty lists for train and test indices
train_indices = []
test_indices = []

train_indices, test_indices = train_test_split(post_id_set, test_size=0.1, random_state=1)  # Adjust the test_size as needed


sentence_triplets_df = pd.read_csv('Datasets/Kialo/argument_triplets_branchleaf.csv')

print(f"original length: {len(sentence_triplets_df)}")
sentence_triplets_df = sentence_triplets_df.dropna()
print(f"length after droping nulls: {len(sentence_triplets_df)}")
sentence_triplets_df.drop_duplicates(inplace=True)
print(f"length after dropping duplicates: {len(sentence_triplets_df)}")
sentence_triplets_df = sentence_triplets_df[(sentence_triplets_df['text_anchor'].progress_apply(lambda x: len(x.split(' ')))>3) & (sentence_triplets_df['text_pro'].progress_apply(lambda x: len(x.split(' ')))>3) & (sentence_triplets_df['text_con'].progress_apply(lambda x: len(x.split(' ')))>3)]
print(f"length after removing short sentences: {len(sentence_triplets_df)}")
# Splitting into train and test sets
# Group the DataFrame by "post_id"


# Split the original DataFrame using the train and test indices
sentence_triplets_train_df = sentence_triplets_df[sentence_triplets_df["post_id"].isin(train_indices)]
sentence_triplets_train_df = sentence_triplets_train_df[sentence_triplets_train_df["lowest_cosine_similarity"]>removal_cutoff_triplet]

sentence_triplets_test_df = sentence_triplets_df[sentence_triplets_df["post_id"].isin(test_indices)]


print(f"length of train-set after split and removal of below {int(removal_cutoff_triplet*100)}% cosine-similarity: {len(sentence_triplets_train_df)}")
print(f"length of test-set: {len(sentence_triplets_test_df)}")

sentence_triplets_train_df.reset_index(drop=True,inplace=True)
sentence_triplets_df = None
sentence_triplets_test_df = None

print(sentence_triplets_train_df.head(2),flush=True)



# Custom dataset for your DataFrame
class CustomDataset_Triplet(Dataset):
    def __init__(self, df):
        self.text_a = df['text_anchor']
        self.text_p = df['text_pro']
        self.text_c = df['text_con']

    def __len__(self):
        return max([len(self.text_a),len(self.text_p),len(self.text_c)])

    def __getitem__(self, idx):
        text_a = self.text_a[idx]
        text_p = self.text_p[idx]
        text_c = self.text_c[idx]
        return text_a, text_p, text_c






sentence_pairs_df = pd.concat([pd.read_csv('Datasets/Kialo/argument_pairs_branchleaf.csv'),pd.read_csv('Datasets/Kialo/argument_pairs_leafleaf.csv')],ignore_index=True)
sentence_pairs_df = sentence_pairs_df[(sentence_pairs_df['stance']=='Pro') | (sentence_pairs_df['stance']=='Con')]
sentence_pairs_df['stance_numeric'] = sentence_pairs_df['stance'].map({'Pro':1,'Con':0})
print(f"original length: {len(sentence_pairs_df)}")
sentence_pairs_df = sentence_pairs_df.dropna()
print(f"length after droping nulls: {len(sentence_pairs_df)}")
sentence_pairs_df = sentence_pairs_df[sentence_pairs_df['sentence_i'] != sentence_pairs_df['sentence_j']]
sentence_pairs_df['sorted_sentences'] = sentence_pairs_df.progress_apply(lambda row: tuple(sorted([row['sentence_i'], row['sentence_j']])), axis=1)
sentence_pairs_df.drop_duplicates(subset='sorted_sentences',inplace=True)
sentence_pairs_df.drop(columns=['sorted_sentences'],inplace=True)
print(f"length after dropping duplicates: {len(sentence_pairs_df)}")
sentence_pairs_df = sentence_pairs_df[(sentence_pairs_df['sentence_i'].progress_apply(lambda x: len(x.split(' ')))>3) & (sentence_pairs_df['sentence_j'].progress_apply(lambda x: len(x.split(' ')))>3)]
print(f"length after removing short sentences: {len(sentence_pairs_df)}")
# Splitting into train and test sets

# Split the original DataFrame using the train and test indices
sentence_pairs_train_df = sentence_pairs_df[sentence_pairs_df["post_id"].isin(train_indices)]
sentence_pairs_train_df = sentence_pairs_train_df[sentence_pairs_train_df["cosine_similarity"]>removal_cutoff_contrastive]

sentence_pairs_test_df = sentence_pairs_df[sentence_pairs_df["post_id"].isin(test_indices)]


print(f"length of train-set after split and removal of below {int(removal_cutoff_contrastive*100)}% cosine-similarity: {len(sentence_pairs_train_df)}")
print(f"length of test-set: {len(sentence_pairs_test_df)}")

sentence_pairs_train_df.reset_index(drop=True,inplace=True)
sentence_pairs_df = None
sentence_pairs_test_df = None

print(sentence_pairs_train_df.head(2),flush=True)



# Custom dataset for your DataFrame
class CustomDataset_Contrastive(Dataset):
    def __init__(self, df):
        self.sentences_a = df['sentence_i']
        self.sentences_b = df['sentence_j']
        self.stance_labels = df['stance_numeric']

    def __len__(self):
        return len(self.sentences_a)

    def __getitem__(self, idx):
        sentence_a = self.sentences_a[idx]
        sentence_b = self.sentences_b[idx]
        stance_label = self.stance_labels[idx]
        return sentence_a, sentence_b, stance_label





# Define your custom stance-aware loss function
class SiameseNetworkMPNet(nn.Module):
    def __init__(self, model_name, tokenizer, normalize=True):
        super(SiameseNetworkMPNet, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **inputs):
        model_output = self.model(**inputs)
        attention_mask = inputs['attention_mask']
        last_hidden_states = model_output.last_hidden_state  # First element of model_output contains all token embeddings
        embeddings = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), 1) / torch.clamp(attention_mask.sum(1, keepdim=True), min=1e-9) # mean_pooling
        if self.normalize:
            embeddings = F.layer_norm(embeddings, embeddings.shape[1:])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


    

    
# Training loop
total_epochs = 4  # should be a large number
# margin_x = 0.6
accumulation_steps = 10  # Gradient accumulation step
batch_size = 64
all_margins = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]


for margin_x in all_margins:
    
    print(f'\nbeginning for parameters:\n  Margin = {margin_x}\n  Batch Size = {batch_size}\n  Accumulation Steps = {accumulation_steps}\n  Total Epochs = {total_epochs}\n\n')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device,flush=True)
    torch.cuda.empty_cache()

    model_name_x = 'sentence-transformers/all-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_x)
    model = SiameseNetworkMPNet(model_name=model_name_x, tokenizer=tokenizer)
    # # add LoRA adaptor
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.to(device)

    dataset = CustomDataset_Triplet(sentence_triplets_train_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_info_df = pd.DataFrame()
    
    triplet_loss = nn.TripletMarginLoss(margin=margin_x)#, p=2, eps=1e-7)

    # Instantiate optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-5)#, correct_bias=True)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=total_epochs,
    )

    # Now we train the model
    max_grad_norm = 1

    if len(sentence_triplets_train_df) % batch_size != 0:
        n_batch_per_epoch = (len(sentence_triplets_train_df) // batch_size) + 1
    if len(sentence_triplets_train_df) % batch_size == 0:
        n_batch_per_epoch = len(sentence_triplets_train_df) // batch_size

    # log_interval = len(dataloader) // 4  # Print loss every 10% of an epoch

    model.train()


    for epoch in range(1,int(total_epochs/2)+1):
        # Create the progress bar
        print(f'Epoch: {epoch}')
        progress_bar = tqdm(total=n_batch_per_epoch, desc='Training', leave=True)
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            text_a = tokenizer(batch[0], return_tensors="pt", max_length=128, truncation=True, padding="max_length")
            text_p = tokenizer(batch[1], return_tensors="pt", max_length=128, truncation=True, padding="max_length")
            text_c = tokenizer(batch[2], return_tensors="pt", max_length=128, truncation=True, padding="max_length")

            embeddings_a = model(**text_a.to(device))
            embeddings_p = model(**text_p.to(device))
            embeddings_c = model(**text_c.to(device))
            

            loss = triplet_loss(embeddings_a, embeddings_p, embeddings_c)

            # Perform gradient accumulation
            loss = loss / accumulation_steps  # Normalize the loss to account for accumulation steps
            loss.backward()

            running_loss += loss.item()

            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient accumulation complete for this accumulation step, perform parameter update
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

                # Update the progress bar after each parameter update
                progress_bar.update(accumulation_steps)
                avg_loss = running_loss / (batch_idx+1)
                progress_bar.set_postfix({'Type':'Triplet','Margin':margin_x,'Epoch': epoch,'Avg-Loss': f'{avg_loss*1000:.2f}','Loss': f'{loss.item()*1000:.2f}'})
                loss_info_df = pd.concat([loss_info_df, pd.DataFrame([{'Loss-Type':'Triplet','Margin':margin_x,'Epoch': epoch,'Average-Loss': avg_loss,'MiniBatch-Loss':loss.item()}])], ignore_index=True)

        # If the last batch has not completed the accumulation step, perform parameter update
        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            progress_bar.update((batch_idx + 1) % accumulation_steps)  # Update progress bar for remaining updates
            progress_bar.set_postfix({'Type':'Triplet','Margin':margin_x,'Epoch': epoch,'Avg-Loss': f'{avg_loss*1000:.2f}','Loss': f'{loss.item()*1000:.2f}'})  # Update loss for the last batch of the epoch
            loss_info_df = pd.concat([loss_info_df, pd.DataFrame([{'Loss-Type':'Triplet','Margin':margin_x,'Epoch': epoch,'Average-Loss': avg_loss,'MiniBatch-Loss':loss.item()}])], ignore_index=True)
        models_dir = '/csg_nas/Vahid/Datasets/StanceAwareSBERT/Models/'
        model_save_path = f'{models_dir}MPNet_contriplet_removal_{int(removal_cutoff_contrastive*100)}_margin_{int(margin_x*100)}_epoch_{epoch}'
        model.save_pretrained(model_save_path)
    #         torch.save(model.state_dict(), model_save_path)
        print(f'\n\nModel saved in:   {model_save_path}\n\n',flush=True)

    print(f'\nbeginning for parameters:\n  Margin = {margin_x}\n  Batch Size = {batch_size}\n  Accumulation Steps = {accumulation_steps}\n  Total Epochs = {total_epochs}\n\n')
    




    dataset = CustomDataset_Contrastive(sentence_pairs_train_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Now we train the model

    if len(sentence_pairs_train_df) % batch_size != 0:
        n_batch_per_epoch = (len(sentence_pairs_train_df) // batch_size) + 1
    if len(sentence_pairs_train_df) % batch_size == 0:
        n_batch_per_epoch = len(sentence_pairs_train_df) // batch_size

    # log_interval = len(dataloader) // 4  # Print loss every 10% of an epoch

    for epoch in range(1,int(total_epochs/2)+1):
        # Create the progress bar
        print(f'Epoch: {epoch}')
        progress_bar = tqdm(total=n_batch_per_epoch, desc='Training', leave=True)
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            text1 = tokenizer(batch[0], return_tensors="pt", max_length=128, truncation=True, padding="max_length")
            text2 = tokenizer(batch[1], return_tensors="pt", max_length=128, truncation=True, padding="max_length")

            embeddings_a = model(**text1.to(device))
            embeddings_b = model(**text2.to(device))

            stance_labels = batch[2].type(torch.float32).to(device)

            cosine_distance_x = 1 - nn.functional.cosine_similarity(embeddings_a, embeddings_b)
            loss = torch.mean(0.5 * (stance_labels * cosine_distance_x.pow(2) + (1 - stance_labels) * F.relu(margin_x*2 - cosine_distance_x).pow(2)))

            # Perform gradient accumulation
            loss = loss / accumulation_steps  # Normalize the loss to account for accumulation steps
            loss.backward()

            running_loss += loss.item()

            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient accumulation complete for this accumulation step, perform parameter update
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

                # Update the progress bar after each parameter update
                progress_bar.update(accumulation_steps)
                avg_loss = running_loss / (batch_idx+1)
                progress_bar.set_postfix({'Type':'Contrastive','Margin':margin_x*100,'Epoch': 2+epoch,'Avg-Loss': f'{avg_loss*1000:.2f}','Loss': f'{loss.item()*1000:.2f}'})
                loss_info_df = pd.concat([loss_info_df, pd.DataFrame([{'Loss-Type':'Contrastive','Margin':margin_x,'Epoch': 2+epoch,'Average-Loss': avg_loss,'MiniBatch-Loss':loss.item()}])], ignore_index=True)

            # ... (previous code)

        # If the last batch has not completed the accumulation step, perform parameter update
        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            progress_bar.update((batch_idx + 1) % accumulation_steps)  # Update progress bar for remaining updates
            progress_bar.set_postfix({'Loss-Type':'Contrastive','Margin':margin_x,'Epoch': 2+epoch,'Average-Loss': f'{avg_loss*1000:.2f}','MiniBatch-Loss': f'{loss.item()*1000:.2f}'})  # Update loss for the last batch of the epoch
            loss_info_df = pd.concat([loss_info_df, pd.DataFrame([{'Loss-Type':'Contrastive','Margin':margin_x,'Epoch': 2+epoch,'Average-Loss': avg_loss,'MiniBatch-Loss':loss.item()}])], ignore_index=True)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        models_dir = 'Models/'
        model_save_path = f'{models_dir}MPNet_contriplet_removal_{int(removal_cutoff_contrastive*100)}_margin_{int(margin_x*100)}_epoch_{2 + epoch}'
        model.save_pretrained(model_save_path)
    #         torch.save(model.state_dict(), model_save_path)
        print(f'\n\nModel saved in:   {model_save_path}\n\n',flush=True)
    loss_info_df.to_csv(f'{models_dir}MPNet_contriplet_removal_{int(removal_cutoff_contrastive*100)}_margin_{int(margin_x*100)}_LossInfo.csv',index=None)
    #     progress_bar.refresh()  # Refresh the progress bar

    progress_bar.close()
