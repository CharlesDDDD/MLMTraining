import torch
from transformers import LineByLineTextDataset, Trainer, TrainingArguments, BertTokenizer, BertForMaskedLM, \
    DataCollatorForLanguageModeling

# setting device for transformers
torch.cuda.set_device(1)
print(torch.cuda.current_device())

tokenizer = BertTokenizer.from_pretrained('xlm-roberta-base')
model = BertForMaskedLM.from_pretrained('xlm-roberta-base')

# initialize the training argument
training_args = TrainingArguments(
    output_dir='models',  # output directory to where save model checkpoint
    evaluation_strategy="steps",  # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=2,  # number of training epochs, feel free to tweak
    per_device_train_batch_size=16,  # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,  # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)
# initialize data_collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# initialize datasets
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='data/xnli/xnli-all.txt',
    block_size=128,
)

# initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# training procedure
trainer.train()

# Save
trainer.save_model('./models/xlmr_base_xnli')
tokenizer.save_pretrained('./models/xlmr_base_xnli')
print('Finished training all... at ./models/xlmr_base_xnli')
