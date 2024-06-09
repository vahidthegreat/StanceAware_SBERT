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

removal_cutoff = 0.5
removal_cutoff = 0.0


# from torchview import draw_graph
# from torchsummary import summary

from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type='nf4',
#         bnb_4bit_compute_dtype='float16',
#         bnb_4bit_use_double_quant=False
#         )



peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q", "k","v","o"],
        bias="none",
        lora_dropout=0.05, # Conventional
#         task_type="CAUSAL_LM",
#         modules_to_save = ["lm_head", "embed_tokens"]   # because we added new tokens
        )



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
# sentence_pairs_df, sentence_pairs_test_df = train_test_split(sentence_pairs_df, test_size=0.1, random_state=1)
# Group the DataFrame by "post_id"
meta_data = pd.read_excel('Datasets/Kialo_MetaData_all.xlsx')
meta_data = meta_data[meta_data['language']=='en']
post_id_set = list(set(list(meta_data["post_id"])))
# Initialize empty lists for train and test indices
train_indices = []
test_indices = []


train_indices, test_indices = train_test_split(post_id_set, test_size=0.1, random_state=1)  # Adjust the test_size as needed

# Split the original DataFrame using the train and test indices
sentence_pairs_train_df = sentence_pairs_df[sentence_pairs_df["post_id"].isin(train_indices)]
sentence_pairs_train_df = sentence_pairs_train_df[sentence_pairs_train_df["cosine_similarity"]>removal_cutoff]

sentence_pairs_test_df = sentence_pairs_df[sentence_pairs_df["post_id"].isin(test_indices)]


print(f"length of train-set after split and removal of below {int(removal_cutoff*100)}% cosine-similarity: {len(sentence_pairs_train_df)}")
print(f"length of test-set: {len(sentence_pairs_test_df)}")

sentence_pairs_train_df.reset_index(drop=True,inplace=True)
sentence_pairs_df = None
sentence_pairs_test_df = None

print(sentence_pairs_train_df.head(2),flush=True)









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

# Custom dataset for your DataFrame
class CustomDataset(Dataset):
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



    dataset = CustomDataset(sentence_pairs_train_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_info_df = pd.DataFrame()

    # Instantiate optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-5)#, correct_bias=True)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=total_epochs,
    )

    # Now we train the model
    max_grad_norm = 1

    if len(sentence_pairs_train_df) % batch_size != 0:
        n_batch_per_epoch = (len(sentence_pairs_train_df) // batch_size) + 1
    if len(sentence_pairs_train_df) % batch_size == 0:
        n_batch_per_epoch = len(sentence_pairs_train_df) // batch_size

    # log_interval = len(dataloader) // 4  # Print loss every 10% of an epoch

    model.train()


    for epoch in range(1,total_epochs+1):
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
            loss = torch.mean(0.5 * (stance_labels * cosine_distance_x.pow(2) + (1 - stance_labels) * F.relu(2*margin_x - cosine_distance_x).pow(2)))

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
                progress_bar.set_postfix({'Epoch': epoch,'Average-Loss': f'{avg_loss*1000:.2f}','MiniBatch-Loss': f'{loss.item()*1000:.2f}'})
                loss_info_df = pd.concat([loss_info_df, pd.DataFrame([{'Epoch': epoch,'Average-Loss': avg_loss,'MiniBatch-Loss':loss.item()}])], ignore_index=True)

            # ... (previous code)

        # If the last batch has not completed the accumulation step, perform parameter update
        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            progress_bar.update((batch_idx + 1) % accumulation_steps)  # Update progress bar for remaining updates
            progress_bar.set_postfix({'Epoch': epoch,'Average-Loss': f'{avg_loss*1000:.2f}','MiniBatch-Loss': f'{loss.item()*1000:.2f}'})  # Update loss for the last batch of the epoch
            loss_info_df = pd.concat([loss_info_df, pd.DataFrame([{'Epoch': epoch,'Average-Loss': avg_loss,'MiniBatch-Loss':loss.item()}])], ignore_index=True)
        models_dir = 'Datasets/StanceAwareSBERT/Models/'
        model_save_path = f'{models_dir}MPNet_contrastive_removal_{int(removal_cutoff*100)}_margin_{int(margin_x*100)}_epoch_{epoch}'
        model.save_pretrained(model_save_path)
    #         torch.save(model.state_dict(), model_save_path)
        print(f'\n\nModel saved in:   {model_save_path}\n\n',flush=True)
    loss_info_df.to_csv(f'{models_dir}MPNet_contrastive_LoRA_removal_{int(removal_cutoff*100)}_margin_{int(margin_x*100)}_LossInfo.csv',index=None)
    #     progress_bar.refresh()  # Refresh the progress bar

    progress_bar.close()
