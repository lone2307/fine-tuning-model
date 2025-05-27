from curses import raw
import tokenize
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from settings import *



def dataLoader(dataset):
    print("Loading and processing dataset...")


    # Loading dataset and put as user - assistant prompt
    if dataset == "oasst1":
        raw_data = load_dataset("OpenAssistant/oasst1", split="train")
        english = raw_data.filter(lambda x: x["lang"] == "en" and x["role"] in {"prompter", "assistant"})

        parent_map = {}
        for ex in english:
            pid = ex["parent_id"]
            if pid:
                parent_map.setdefault(pid, []).append(ex)

        examples = []
        for ex in english:
            if ex["role"] == "prompter":
                for child in parent_map.get(ex["message_id"], []):
                    if child["role"] == "assistant":
                        examples.append({
                            "prompt": ex["text"],
                            "response": child["text"]
                        })

    elif dataset == "WildChat":
        raw_data = load_dataset("allenai/WildChat-1M", split="train")
        english = raw_data.filter(lambda x: x["language"] == "English")

        examples = []
        for ex in english:
            message = ex["conversation"]
            for convo in range(0, len(message), 2):
                examples.append({
                    "prompt": message[convo]["content"],
                    "response": message[convo + 1]["content"]
                })

    # Format data for instruction tuning
    def format_example(example):
        return {"text": f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['response']}"}

    formatted_data = list(map(format_example, examples))

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = Dataset.from_list(formatted_data).map(tokenize, batched=True)

    return tokenized_dataset