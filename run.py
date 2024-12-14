import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import torch


NUM_PREPROCESSING_WORKERS = 2


def main():
    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA is not available. Please ensure a compatible GPU and drivers are installed.")
    # print("CUDA is available. Using device:", torch.cuda.get_device_name(0))

    # # Ensure all operations use the GPU
    # device = torch.device("cuda")

    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    
    # added training optimizations
    # argp.add_argument('--learning_rate', type=float, default=2e-5)
    # argp.add_argument('--warmup_steps', type=int, default=1000)
    # argp.add_argument('--gradient_accumulation_steps', type=int, default=2)
    # argp.add_argument('--weight_decay', type=float, default=0.01)

    
    training_args, args = argp.parse_args_into_dataclasses()

    def get_anli_dataset():
        """Load ANLI dataset"""
        # Load the entire ANLI dataset
        dataset = datasets.load_dataset('anli') 
        # print(dataset.keys())
        
        # Create combined training set
        train_datasets = [dataset[f'train_r{i}'] for i in range(1, 4)]  # Collect all train rounds
        validation_datasets = [dataset[f'dev_r{i}'] for i in range(1, 4)] # Collect all dev rounds
    
        combined_train = datasets.concatenate_datasets(train_datasets)
        combined_validation = datasets.concatenate_datasets(validation_datasets)

        return datasets.DatasetDict({
            'train': combined_train,
            'validation': combined_validation
        })

    def create_mixed_dataset(snli_dataset, anli_dataset, snli_ratio=0.3):
        """Mix SNLI and ANLI datasets with given ratio of SNLI examples"""
        snli_train = snli_dataset['train'].shuffle(seed=42)
        anli_train = anli_dataset['train']
        
        # Select subset of SNLI
        num_snli = int(len(anli_train) * snli_ratio / (1 - snli_ratio))
        snli_subset = snli_train.select(range(num_snli))
        
        # Combine datasets
        combined_train = datasets.concatenate_datasets([snli_subset, anli_train])
        return combined_train.shuffle(seed=42)
    
    def get_hans_dataset():
        """Load HANS dataset"""
        # Load the dataset from the Hugging Face hub
        hans_dataset = datasets.load_dataset("hans")
        return hans_dataset
    
    def prepare_train_dataset_hans(examples, tokenizer, max_length):
    # Tokenize the dataset (ensure that it matches the structure of SNLI)
        return tokenizer(examples['premise'], examples['hypothesis'], padding=True, truncation=True, max_length=max_length)

    def prepare_eval_dataset_hans(examples, tokenizer, max_length):
        # Similar preprocessing for evaluation
        return tokenizer(examples['premise'], examples['hypothesis'], padding=True, truncation=True, max_length=max_length)
    
    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    # Replace the dataset loading section with:
    default_datasets = {
        'qa': ('squad',), 
        'nli': ('snli',),
        'anli': ('anli',)
    }
    if args.dataset == 'hans':
        hans_dataset = get_hans_dataset()
        dataset = {
            'train': hans_dataset['train'],
            'validation': hans_dataset['validation']
        }
        eval_split = 'validation'  # Use validation split from HANS dataset

        prepare_train_dataset = lambda exs: prepare_train_dataset_hans(exs, tokenizer, args.max_length)
        prepare_eval_dataset = lambda exs: prepare_eval_dataset_hans(exs, tokenizer, args.max_length)
    elif args.dataset == 'anli':
        anli_dataset = get_anli_dataset()
        snli_dataset = datasets.load_dataset('snli')
        snli_dataset = snli_dataset.filter(lambda ex: ex['label'] != -1)
        mixed_train = create_mixed_dataset(snli_dataset, anli_dataset)
        dataset = datasets.DatasetDict({
            'train': mixed_train,
            'validation': anli_dataset['validation']
        })
        eval_split = 'validation'  # Set eval_split for ANLI dataset
    elif args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        eval_split = 'train' 
    else:
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else default_datasets[args.task]
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        dataset = datasets.load_dataset(*dataset_id)
        
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    # model = AutoModelForSequenceClassification.from_pretrained("./trained_model/") 
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # tokenizer = AutoTokenizer.from_pretrained("./trained_model/")

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    # comment out this if statement if it gives this error "RuntimeError: CUDA error: device-side assert triggered" 
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )
    # if training_args.do_eval:
    #     eval_dataset = dataset[eval_split]
    #     if args.max_eval_samples:
    #         eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    #     # Load contrast examples
    #     with open('contrast_examples.jsonl', 'r', encoding='utf-8') as f:
    #         contrast_examples = [json.loads(line) for line in f]
        
    #     # Convert to Hugging Face Dataset
    #     contrast_dataset = datasets.Dataset.from_list(contrast_examples)

    #     # Combine with eval dataset
    #     eval_dataset = datasets.concatenate_datasets([eval_dataset, contrast_dataset])

    #     eval_dataset_featurized = eval_dataset.map(
    #         prepare_eval_dataset,
    #         batched=True,
    #         num_proc=NUM_PREPROCESSING_WORKERS,
    #         remove_columns=eval_dataset.column_names
    #     )


    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = evaluate.load('squad')   # datasets.load_metric() deprecated
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    # def compute_metrics_and_store_predictions(eval_preds):
    #     nonlocal eval_predictions
    #     eval_predictions = eval_preds
    #     return compute_metrics(eval_preds)

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds

        # Compute metrics
        metrics = compute_metrics(eval_preds)

        # Open the output file for storing failed tests
        failed_tests_file = os.path.join(training_args.output_dir, 'model_failed_tests.txt')
        with open(failed_tests_file, 'w', encoding='utf-8') as f:
            for i, example in enumerate(eval_dataset):
                # Check if the model's prediction is wrong
                if args.task == 'qa':
                    # For question answering, compare predicted_answer with the actual answer
                    predicted_answer = eval_predictions.predictions[i]
                    ground_truth = example['answers']['text'][0]  # Assuming a single answer is provided

                    if predicted_answer != ground_truth:
                        # Write the failed example to the file
                        failed_example = {
                            'question': example['question'],
                            'context': example['context'],
                            'predicted_answer': predicted_answer,
                            'ground_truth': ground_truth
                        }
                        f.write(json.dumps(failed_example) + '\n')
                elif args.task == 'nli':
                    # For NLI, compare predicted label with actual label
                    predicted_label = int(eval_predictions.predictions[i].argmax())
                    ground_truth = example['label']

                    if predicted_label != ground_truth:
                        # Write the failed example to the file
                        failed_example = {
                            'premise': example['premise'],
                            'hypothesis': example['hypothesis'],
                            'predicted_label': predicted_label,
                            'ground_truth': ground_truth
                        }
                        f.write(json.dumps(failed_example) + '\n')
                    # print("label: ", ground_truth, "predicted_answer: ", predicted_label)
        
        return metrics


    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')

if __name__ == "__main__":
    main()
