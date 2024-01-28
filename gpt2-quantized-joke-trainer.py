import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import os
import csv
from tqdm import tqdm
import numpy as np

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')


class JokesDataset(Dataset):
    def __init__(self, jokes_dataset_path='data/', train_n=231658):
        super().__init__()

        short_jokes_path = os.path.join(jokes_dataset_path, 'shortjokes.csv')

        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"

        with open(short_jokes_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            x = 0
            for row in csv_reader:
                joke_str = f"JOKE:{row[1]}{self.end_of_text_token}"
                self.joke_list.append(joke_str)
        self.joke_list = self.joke_list[:train_n]

    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]


def train(dataloader, model, model_name, optimizer, scheduler, device):

    tmp_jokes_tens = None
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0

    if not os.path.exists(model_name):
        os.mkdir(model_name)

    #model = model.to(device)                   #`.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
    model.train()

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch} started" + '=' * 30)

        for idx, joke in tqdm(enumerate(dataloader)):

            #################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
            joke_tens = torch.tensor(tokenizer.encode(joke[0])).unsqueeze(0).to(device)
            # Skip sample from dataset if it is longer than MAX_SEQ_LEN
            if joke_tens.size()[1] > MAX_SEQ_LEN:
                continue

            # The first joke sequence in the sequence
            if not torch.is_tensor(tmp_jokes_tens):
                tmp_jokes_tens = joke_tens
                continue
            else:
                # The next joke does not fit in so we process the sequence and leave the last joke
                # as the start for next sequence
                if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > MAX_SEQ_LEN:
                    work_jokes_tens = tmp_jokes_tens
                    tmp_jokes_tens = joke_tens
                else:
                    # Add the joke to sequence, continue and try to add more
                    tmp_jokes_tens = torch.cat([tmp_jokes_tens, joke_tens[:, 1:]], dim=1)
                    continue
            ################## Sequence ready, process it trough the model ##################

            outputs = model(work_jokes_tens, labels=work_jokes_tens)
            loss, logits = outputs[:2]
            loss.backward()
            sum_loss = sum_loss + loss.detach().data

            proc_seq_count = proc_seq_count + 1
            if proc_seq_count == BATCH_SIZE:
                proc_seq_count = 0
                batch_count += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 100:
                print(f"sum loss {sum_loss}")
                batch_count = 0
                sum_loss = 0.0

        # Store the model after each epoch to compare the performance of them
        torch.save(model.state_dict(), os.path.join(model_name, f"{model_name}.pt"))
        model.save_pretrained(model_name)
        tokenizer.save_pretrained(model_name)

    return model, tokenizer



#------------------------------------------------------- EVALUATION/GENERATION ---------------------------------------------

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def eval(model_name, model, tokenizer):

    model_path = os.path.join(model_name, f"{model_name}.pt")
    model.load_state_dict(torch.load(model_path))

    jokes_output_file_path = f'generated_jokes.txt'
    model.eval()
    if os.path.exists(jokes_output_file_path):
        os.remove(jokes_output_file_path)

    joke_num = 0
    with torch.no_grad():
        for joke_idx in tqdm(range(5)):

            joke_finished = False

            cur_ids = torch.tensor(tokenizer.encode("JOKE:")).unsqueeze(0).to(device)

            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0, -1],
                                               dim=0)  # Take the first(from only one in this case) batch and the last predicted embedding
                if i < 3:
                    n = 20
                else:
                    n = 3
                next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(),
                                                n=n)  # Randomly(from the topN probability distribution) select the next word
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id],
                                    dim=1)  # Add the last word to the running sequence

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    joke_finished = True
                    break

            if joke_finished:
                joke_num = joke_num + 1

                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list, skip_special_tokens=True)
                print(f"Jokes #{joke_idx}: {output_text}")
                with open(jokes_output_file_path, 'a') as f:
                    f.write(f"{output_text} \n\n")



def eval2(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = tokenizer.apply_chat_template('JOKE: ', add_generation_prompt=True, tokenize=False)

    # Create pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer
    )

    # Generate text
    sequences = pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        max_length=200,
    )
    print(sequences[0]['generated_text'])

def upload_to_huggingface(model_name, model, tokenizer):
    model.push_to_hub(model_name, use_temp_dir=False, token=hf_token)
    tokenizer.push_to_hub(model_name, use_temp_dir=False, token=hf_token)


if __name__ == "__main__":

    # environment variables
    #hf_token = os.environ['hf_token']
    hf_token = "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"

    # hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WARMUP_STEPS = 5000
    MAX_SEQ_LEN = 400
    model_name = 'gpt2-quantized-jokes'
    #n_train_data = 1000


    # load dataset
    dataset = JokesDataset()
    jokes_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    # load model
    device = 'cpu' if torch.cuda.is_available() else 'cuda'
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2',
                                            device_map='auto',
                                            load_in_8bit=True
                                            )
    #model = model.to(device)   # `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=EPOCHS, last_epoch=-1)


    trained_model, trained_tokenizer = train(dataloader=jokes_loader, model=model, model_name=model_name, optimizer=optimizer, scheduler=scheduler, device=device)

    upload_to_huggingface(model_name, trained_model, trained_tokenizer)

    #model.to(device)
    #eval(model_name, model, tokenizer)
    eval2(model_name, trained_tokenizer)

    print(f"Quantized GPT-2 model size: {model.get_memory_footprint() / 1e6:,} Mb")
