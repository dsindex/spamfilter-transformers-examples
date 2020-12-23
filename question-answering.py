import sys
import os
import argparse
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
from tqdm.auto import tqdm
import collections

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fileHandler = logging.FileHandler('./train.log')
logger.addHandler(fileHandler)

def prepare_datasets(opt):
    datasets = load_dataset('squad_v2' if opt.squad_v2 else 'squad') 
    print(datasets['train'][0])

    logger.info("datasets ready")

    return datasets

def preprocess_datasets(opt, datasets, tokenizer):

    pad_on_right = tokenizer.padding_side == 'right'
    max_length = opt.max_length
    doc_stride = opt.doc_stride
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)

    print(tokenized_datasets['validation'][0])
    logger.info("preprocessing datasets done")

    return tokenized_datasets

def preprocess_validation_datasets(opt, datasets, tokenizer):

    pad_on_right = tokenizer.padding_side == 'right'
    max_length = opt.max_length
    doc_stride = opt.doc_stride

    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    validation_features = datasets["validation"].map(prepare_validation_features, batched=True, remove_columns=datasets["validation"].column_names) 
    print(validation_features[0])
    logger.info("preprocessing validation datasets done")
    
    return validation_features

def postprocess_qa_predictions(opt, examples, features, raw_predictions, tokenizer, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not opt.squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions

def get_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', type=str, default='test-squad')
    parser.add_argument('--squad_v2', action='store_true',
                        help="Set this flag for squad_v2.")
    parser.add_argument('--metric_for_best_model', type=str, default='eval_loss')
    parser.add_argument('--bert_model_name_or_path', type=str, default='distilbert-base-uncased')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=384, help="The maximum length of a feature (question and context).")
    parser.add_argument('--doc_stride', type=int, default=128, help="The authorized overlap between two part of the context when splitting it is needed.")
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--eval_steps', default=500, type=int)
    parser.add_argument('--save_steps', default=1000, type=int)
    parser.add_argument('--hp_search_ray', action='store_true',
                        help="Set this flag to use hyper-parameter search by Ray.")
    parser.add_argument('--hp_trials', default=10, type=int)
    parser.add_argument('--hp_n_jobs', default=1, type=int)
    parser.add_argument('--hp_dashboard_port', default=8080, type=int)

    opt = parser.parse_args()
    return opt

def main():
    opt = get_params()

    datasets = prepare_datasets(opt)

    tokenizer = AutoTokenizer.from_pretrained(opt.bert_model_name_or_path, use_fast=True)

    tokenized_datasets = preprocess_datasets(opt, datasets, tokenizer)

    data_collator = default_data_collator

    # prepare model
    def model_init():
        return AutoModelForQuestionAnswering.from_pretrained(opt.bert_model_name_or_path, return_dict=True)
    model = model_init()
    logger.info("classification model ready")

    # prepare trainer and train
    if opt.hp_search_ray:
        args = TrainingArguments(
            opt.task,
            do_train=True,
            do_eval=True,
            learning_rate=opt.learning_rate,
            per_device_train_batch_size=opt.per_device_train_batch_size,
            per_device_eval_batch_size=opt.per_device_eval_batch_size,
            num_train_epochs=opt.num_train_epochs,
            weight_decay=opt.weight_decay,
            warmup_steps=opt.warmup_steps,
            evaluate_during_training=True,
            eval_steps=opt.eval_steps,
            save_steps=opt.save_steps,
            disable_tqdm=True
        )
        trainer = Trainer(
            args=args,
            tokenizer=tokenizer,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            model_init=model_init,
            data_collator=data_collator,
        )
        scheduler = PopulationBasedTraining(
            time_attr='training_iteration',
            metric=opt.metric_for_best_model,
            mode='max',  # it depends metric.compute()
            perturbation_interval=1,
            hyperparam_mutations={
                'weight_decay': tune.uniform(0.0, 0.3),
                'learning_rate': tune.uniform(1e-5, 5e-5),
                'per_device_train_batch_size': [16, 32, 64],
            }) 
        ray.init(dashboard_port=opt.hp_dashboard_port)
        trainer.hyperparameter_search(
            backend='ray',
            direction='maximize',   # it depends metric.compute()
            scheduler=scheduler,
            keep_checkpoints_num=2,
            n_trials=opt.hp_trials, # number of trials
            n_jobs=opt.hp_n_jobs,   # number of parallel jobs, if multiple GPUs
        )
    else:
        args = TrainingArguments(
            opt.task,
            learning_rate=opt.learning_rate,
            per_device_train_batch_size=opt.per_device_train_batch_size,
            per_device_eval_batch_size=opt.per_device_eval_batch_size,
            num_train_epochs=opt.num_train_epochs,
            weight_decay=opt.weight_decay,
            warmup_steps=opt.warmup_steps,
            evaluation_strategy='epoch',
            disable_tqdm=False,
            load_best_model_at_end=True,
            metric_for_best_model=opt.metric_for_best_model,
        )
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        logger.info("training done")

        trainer.save_model("test-squad-trained")
        logger.info("model saved")

        validation_features = preprocess_validation_datasets(opt, datasets, tokenizer)
        raw_predictions = trainer.predict(validation_features)
        logger.info("prediction done")

        validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))

        final_predictions = postprocess_qa_predictions(opt, datasets["validation"], validation_features, raw_predictions.predictions, tokenizer)
        if opt.squad_v2:
            # git clone -b v4.1.1 https://github.com/huggingface/transformers.git
            import os
            # Adapt this to your local environment
            path_to_transformers = "../transformers"
            path_to_qa_examples = os.path.join(path_to_transformers, "examples/question-answering")
            metric = load_metric(os.path.join(path_to_qa_examples, "squad_v2_local"))
            # Uncomment when the fix is merged in master and has been released. 
            #metric = load_metric("squad_v2")
        else:
            metric = load_metric("squad")
       
        if opt.squad_v2:
            formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]

        print(references[0])
        print(formatted_predictions[0])
        '''
        {'id': '56ddde6b9a695914005b9628', 'answers': {'answer_start': [159, 159, 159, 159], 'text': ['France', 'France', 'France', 'France']}}
        {'id': '56ddde6b9a695914005b9628', 'prediction_text': 'France', 'no_answer_probability': 0.0}
        '''
        print(metric.compute(predictions=formatted_predictions, references=references))
        '''
        OrderedDict([('exact', 58.923608186641964), ('f1', 62.25962746847768), ('total', 11873), ('HasAns_exact', 56.15721997300945), ('HasAns_f1', 62.83882539359549), ('HasAns_total', 5928), ('NoAns_exact', 61.682085786375104), ('NoAns_f1', 61.682085786375104), ('NoAns_total', 5945), ('best_exact', 59.1425924366209), ('best_exact_thresh', 0.0), ('best_f1', 62.40809402356971), ('best_f1_thresh', 0.0)])
        '''

if __name__ == '__main__':
    main()
