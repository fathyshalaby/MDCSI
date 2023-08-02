# -*- coding: utf-8 -*-

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch import nn
import torch
from transformers import Trainer
from tqdm import tqdm
import json
from datasets import load_dataset, Dataset, load_metric
from sentence_transformers.losses import (
    CosineSimilarityLoss,
    ContrastiveLoss,
    BatchAllTripletLoss,
    BatchHardTripletLoss,
)
from transformers import DataCollatorWithPadding
import numpy as np
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter
import numpy as np

# set seed
# seed for datasets shuffle

DATASET = None
DOMAIN = None
ID2LABEL = None
SAMPLE_SIZE = None
MODEL = None
WEIGHT = None


def plot_confusion_matrix(y_true, y_pred):
    """This function prints and plots the confusion matrix."""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm.shape)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=list(ID2LABEL.values())
    )

    # save the figure and create as much space between the boxes as possible
    fig, ax = plt.subplots(figsize=(20, 20))
    # rotate the x-axis labels
    disp.plot(ax=ax)
    plt.xticks(rotation=90)
    plt.savefig(
        f"../reports/confs/confusion_matrix_{DATASET}_{DOMAIN}_{SAMPLE_SIZE}_{MODEL}.png"
    )


id2label = {
    "0": "restaurant_reviews",
    "1": "nutrition_info",
    "2": "account_blocked",
    "3": "oil_change_how",
    "4": "time",
    "5": "weather",
    "6": "redeem_rewards",
    "7": "interest_rate",
    "8": "gas_type",
    "9": "accept_reservations",
    "10": "smart_home",
    "11": "user_name",
    "12": "report_lost_card",
    "13": "repeat",
    "14": "whisper_mode",
    "15": "what_are_your_hobbies",
    "16": "order",
    "17": "jump_start",
    "18": "schedule_meeting",
    "19": "meeting_schedule",
    "20": "freeze_account",
    "21": "what_song",
    "22": "meaning_of_life",
    "23": "restaurant_reservation",
    "24": "traffic",
    "25": "make_call",
    "26": "text",
    "27": "bill_balance",
    "28": "improve_credit_score",
    "29": "change_language",
    "30": "no",
    "31": "measurement_conversion",
    "32": "timer",
    "33": "flip_coin",
    "34": "do_you_have_pets",
    "35": "balance",
    "36": "tell_joke",
    "37": "last_maintenance",
    "38": "exchange_rate",
    "39": "uber",
    "40": "car_rental",
    "41": "credit_limit",
    "42": "oos",
    "43": "shopping_list",
    "44": "expiration_date",
    "45": "routing",
    "46": "meal_suggestion",
    "47": "tire_change",
    "48": "todo_list",
    "49": "card_declined",
    "50": "rewards_balance",
    "51": "change_accent",
    "52": "vaccines",
    "53": "reminder_update",
    "54": "food_last",
    "55": "change_ai_name",
    "56": "bill_due",
    "57": "who_do_you_work_for",
    "58": "share_location",
    "59": "international_visa",
    "60": "calendar",
    "61": "translate",
    "62": "carry_on",
    "63": "book_flight",
    "64": "insurance_change",
    "65": "todo_list_update",
    "66": "timezone",
    "67": "cancel_reservation",
    "68": "transactions",
    "69": "credit_score",
    "70": "report_fraud",
    "71": "spending_history",
    "72": "directions",
    "73": "spelling",
    "74": "insurance",
    "75": "what_is_your_name",
    "76": "reminder",
    "77": "where_are_you_from",
    "78": "distance",
    "79": "payday",
    "80": "flight_status",
    "81": "find_phone",
    "82": "greeting",
    "83": "alarm",
    "84": "order_status",
    "85": "confirm_reservation",
    "86": "cook_time",
    "87": "damaged_card",
    "88": "reset_settings",
    "89": "pin_change",
    "90": "replacement_card_duration",
    "91": "new_card",
    "92": "roll_dice",
    "93": "income",
    "94": "taxes",
    "95": "date",
    "96": "who_made_you",
    "97": "pto_request",
    "98": "tire_pressure",
    "99": "how_old_are_you",
    "100": "rollover_401k",
    "101": "pto_request_status",
    "102": "how_busy",
    "103": "application_status",
    "104": "recipe",
    "105": "calendar_update",
    "106": "play_music",
    "107": "yes",
    "108": "direct_deposit",
    "109": "credit_limit_change",
    "110": "gas",
    "111": "pay_bill",
    "112": "ingredients_list",
    "113": "lost_luggage",
    "114": "goodbye",
    "115": "what_can_i_ask_you",
    "116": "book_hotel",
    "117": "are_you_a_bot",
    "118": "next_song",
    "119": "change_speed",
    "120": "plug_type",
    "121": "maybe",
    "122": "w2",
    "123": "oil_change_when",
    "124": "thank_you",
    "125": "shopping_list_update",
    "126": "pto_balance",
    "127": "order_checks",
    "128": "travel_alert",
    "129": "fun_fact",
    "130": "sync_device",
    "131": "schedule_maintenance",
    "132": "apr",
    "133": "transfer",
    "134": "ingredient_substitution",
    "135": "calories",
    "136": "current_location",
    "137": "international_fees",
    "138": "calculator",
    "139": "definition",
    "140": "next_holiday",
    "141": "update_playlist",
    "142": "mpg",
    "143": "min_payment",
    "144": "change_user_name",
    "145": "restaurant_suggestion",
    "146": "travel_notification",
    "147": "cancel",
    "148": "pto_used",
    "149": "travel_suggestion",
    "150": "change_volume",
}

domains_dict = {
    "banking": [
        "freeze_account",
        "routing",
        "pin_change",
        "bill_due",
        "pay_bill",
        "account_blocked",
        "interest_rate",
        "min_payment",
        "bill_balance",
        "transfer",
        "order_checks",
        "balance",
        "spending_history",
        "transactions",
        "report_fraud",
    ],
    "credit_cards": [
        "replacement_card_duration",
        "expiration_date",
        "damaged_card",
        "improve_credit_score",
        "report_lost_card",
        "card_declined",
        "credit_limit_change",
        "apr",
        "redeem_rewards",
        "credit_limit",
        "rewards_balance",
        "application_status",
        "credit_score",
        "new_card",
        "international_fees",
    ],
    "kitchen_and_dining": [
        "food_last",
        "confirm_reservation",
        "how_busy",
        "ingredients_list",
        "calories",
        "nutrition_info",
        "recipe",
        "restaurant_reviews",
        "restaurant_reservation",
        "meal_suggestion",
        "restaurant_suggestion",
        "cancel_reservation",
        "ingredient_substitution",
        "cook_time",
        "accept_reservations",
    ],
    "home": [
        "what_song",
        "play_music",
        "todo_list_update",
        "reminder",
        "reminder_update",
        "calendar_update",
        "order_status",
        "update_playlist",
        "shopping_list",
        "calendar",
        "next_song",
        "order",
        "todo_list",
        "shopping_list_update",
        "smart_home",
    ],
    "auto_and_commute": [
        "current_location",
        "oil_change_when",
        "oil_change_how",
        "uber",
        "traffic",
        "tire_pressure",
        "schedule_maintenance",
        "gas",
        "mpg",
        "distance",
        "directions",
        "last_maintenance",
        "gas_type",
        "tire_change",
        "jump_start",
    ],
    "travel": [
        "plug_type",
        "travel_notification",
        "translate",
        "flight_status",
        "international_visa",
        "timezone",
        "exchange_rate",
        "travel_suggestion",
        "travel_alert",
        "vaccines",
        "lost_luggage",
        "book_flight",
        "book_hotel",
        "carry_on",
        "car_rental",
    ],
    "utility": [
        "weather",
        "alarm",
        "date",
        "find_phone",
        "share_location",
        "timer",
        "make_call",
        "calculator",
        "definition",
        "measurement_conversion",
        "flip_coin",
        "spelling",
        "time",
        "roll_dice",
        "text",
    ],
    "work": [
        "pto_request_status",
        "next_holiday",
        "insurance_change",
        "insurance",
        "meeting_schedule",
        "payday",
        "taxes",
        "income",
        "rollover_401k",
        "pto_balance",
        "pto_request",
        "w2",
        "schedule_meeting",
        "direct_deposit",
        "pto_used",
    ],
    "small_talk": [
        "who_made_you",
        "meaning_of_life",
        "who_do_you_work_for",
        "do_you_have_pets",
        "what_are_your_hobbies",
        "fun_fact",
        "what_is_your_name",
        "where_are_you_from",
        "goodbye",
        "thank_you",
        "greeting",
        "tell_joke",
        "are_you_a_bot",
        "how_old_are_you",
        "what_can_i_ask_you",
    ],
    "meta": [
        "change_speed",
        "user_name",
        "whisper_mode",
        "yes",
        "change_volume",
        "no",
        "change_language",
        "repeat",
        "change_accent",
        "cancel",
        "sync_device",
        "change_user_name",
        "change_ai_name",
        "reset_settings",
        "maybe",
    ],
}


def class_accuracy(predictions, true_labels):
    """
    Calculates the accuracy per class given a list of predicted labels and a list of true labels.

    Args:
        predictions (list): List of predicted labels.
        true_labels (list): List of true labels.

    Returns:
        A dictionary where the keys are the unique classes in the true labels and the values are the
        corresponding accuracy scores.
    """
    # Count the number of occurrences of each unique class in the true labels
    class_counts = Counter(true_labels)

    # Initialize an empty dictionary to store the accuracy scores
    accuracies = {}

    # Loop over the unique classes in the true labels
    for class_name in class_counts.keys():
        # Initialize variables to count the number of correct and total predictions for this class
        correct_predictions = 0
        total_predictions = 0

        # Loop over the predicted and true labels in parallel
        for predicted_label, true_label in zip(predictions, true_labels):
            # If the predicted label matches the true label and the true label is the current class
            if predicted_label == true_label and true_label == class_name:
                correct_predictions += 1

            # If the true label is the current class
            if true_label == class_name:
                total_predictions += 1

        # Calculate the accuracy for this class
        accuracy = correct_predictions / total_predictions

        # Add the accuracy score to the dictionary
        accuracies[ID2LABEL[class_name]] = accuracy

    return accuracies


def compute_metrics(eval_pred):
    y_pred = np.argmax(eval_pred.predictions, axis=1)
    y_true = eval_pred.label_ids
    try:
        plot_confusion_matrix(y_true, y_pred)
    except:
        pass
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_accuracy": class_accuracy(y_pred, y_true),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def compute_metrics_setfit(y_pred, y_true):
    try:
        plot_confusion_matrix(y_true, y_pred)
    except:
        pass
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_accuracy": class_accuracy(y_pred, y_true),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def get_class_weights(y_train):
    class_weights = {}
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = np.max(counts)
    for cls, count in zip(classes, counts):
        class_weights[cls] = max_count / count
    return class_weights


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=WEIGHT)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def ex_transformers(
    model_id,
    dataset,
    sample_size,
    batch_s,
    epochs,
    push_to_hub,
    n_classes,
    domain,
    run,
    full_train=False,
    seed=42,
):
    global SAMPLE_SIZE
    global MODEL
    global WEIGHT
    MODEL = "transformers"
    SAMPLE_SIZE = sample_size
    print("sample_size", sample_size)
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=n_classes
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load the dataset
    if full_train:
        train_dataset = dataset["train"]
    else:
        try:
            train_dataset = sample_dataset(dataset["train"], num_samples=sample_size)
        except:
            train_dataset = sample_dataset(
                dataset["train"], num_samples=sample_size, seed=1234
            )
    test_dataset = dataset["test"]
    print(test_dataset)
    WEIGHT = torch.Tensor(list(get_class_weights(train_dataset["label"]).values())).to("cuda")

    # Preprocess the dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set the training arguments
    model_name = model_id.split("/")[-1]
    training_args = TrainingArguments(
        output_dir=f"../../../../nfs/aishare/Artifacts/Intent/baselines/{model_name}-{domain}-{sample_size}-{batch_s}-{epochs}-{run}-oos",
        evaluation_strategy="no",
        per_device_train_batch_size=batch_s,
        per_device_eval_batch_size=batch_s,
        learning_rate=2e-5,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        greater_is_better=True,
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        push_to_hub=push_to_hub,
        fp16=True,
    )
    # Set the trainer
    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # Train the model
    trainer.train()
    # Evaluate the model
    metrics = trainer.evaluate()
    print(metrics)
    return metrics


def ex_setfit(
    model_id,
    dataset,
    sample_size,
    batch_s=16,
    epochs=5,
    push_to_hub=False,
    n_classes=150,
    full_train=False,
    domain=None,
    run=0,
):
    global SAMPLE_SIZE
    global MODEL
    SAMPLE_SIZE = sample_size
    MODEL = "setfit"
    print("sample_size", sample_size)
    if full_train:
        train_dataset = dataset["train"]
    else:
        try:
            train_dataset = sample_dataset(dataset["train"], num_samples=sample_size)
        except:
            train_dataset = sample_dataset(
                dataset["train"], num_samples=sample_size, seed=1234
            )
    test_dataset = dataset["test"]
    # Load SetFit model from Hub
    model = SetFitModel.from_pretrained(model_id)
    # Create trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_class=CosineSimilarityLoss,
        metric=compute_metrics_setfit,
        batch_size=4,
        num_epochs=epochs,
        learning_rate=1.0e-06,
        warmup_proportion=0.2,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    if push_to_hub:
        trainer.push_to_hub(
            repo_path_or_name=f"fathyshalab/setfit-{domain}-{sample_size}-{batch_s}-{epochs}-{run}-v2"
        )
    return metrics


def clinic():
    global DATASET
    global ID2LABEL
    global DOMAIN
    domains = list(domains_dict.keys())
    domains.append("full")
    approaches = {"setfit": ex_setfit, "transformers": ex_transformers}
    domain_res = {k: {} for k in domains}
    model_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    n_runs = 1
    DATASET = "clinic"

    for domain in tqdm(list(reversed(domains))):
        dataset = load_dataset(f"fathyshalab/clinic-{domain}")
        DOMAIN = domain
        ID2LABEL = {
            k: v
            for k, v in zip(dataset["test"]["label"], dataset["test"]["label_text"])
        }

        # push the dataset to the hub
        # dataset.push_to_hub(f"fathyshalab/clinic-{domain}")
        runs = {r: {} for r in range(n_runs)}

        for run in range(n_runs):
            print(
                f"=======================================Run:{run}======================================="
            )
            approach_res = {k: {} for k in approaches.keys()}
            for approac_name, approach in approaches.items():
                print(
                    f"======================================={approac_name}-{domain}======================================="
                )
                for i in [2, 4, 8, 16, 32, 64, 128]:
                    if domain == "full":
                        approach_res[approac_name][i] = approach(
                            model_id, dataset, i, 4, 5, False, 150, domain, run=run
                        )
                    else:
                        approach_res[approac_name][i] = approach(
                            model_id,
                            dataset,
                            i,
                            4,
                            1,
                            False,
                            n_classes=len(domains_dict[domain]),
                            domain=domain,
                            run=run,
                        )
                approach_res[approac_name]["full"] = approach(
                    model_id,
                    dataset,
                    1000,
                    4,
                    1,
                    False,
                    150,
                    domain,
                    run=run,
                    full_train=True,
                )
                runs[run] = approach_res
        domain_res[domain] = runs
        print(
            "=======================================Done======================================="
        )
        print("domain_res", domain_res)

    with open("clinic-5.json", "w") as f:
        json.dump(domain_res, f)


def reklam24():
    global DATASET
    global ID2LABEL
    global DOMAIN
    domains = [
        "supermaerkte-drogerien",
        "mode-schmuck-zubehoer",
        "moebel-einrichtungshaeuser",
        "finanzen",
        "reisen-tourismus",
        "schoenheit-wellness",
        "unternehmen-verbaende",
        "medizin-gesundheit-pflege",
        "transport-logistik",
        "versicherungen-recht",
        "oeffentlichkeit-soziales",
        "oeffentlicher-verkehr-vermietung",
        "unterhaltung-kultur-freizeit",
        "wasser-strom-gas",
        "haus-reinigung",
        "full"
    ]
    domain_res = {k: {} for k in domains}
    n_runs = 1
    DATASET = "reklam24"
    for dm in tqdm(domains):
        DOMAIN = dm
        print(
            f"=======================================Domain:{dm}======================================="
        )
        dataset = load_dataset(f"fathyshalab/reklamation24_{dm}-v2",use_auth_token=True)
        ID2LABEL = {
            k: v
            for k, v in zip(dataset["test"]["label"], dataset["test"]["label_name"])
        }

        # reset the labels to be starting with zero and no missing label

        model_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        approaches = {"setfit": ex_setfit,"transformers": ex_transformers, }
        domain = dm
        classes = np.unique(dataset["train"]["label"])
        runs = {r: {} for r in range(n_runs)}

        for run in range(n_runs):
            print(
                f"=======================================Run:{run}======================================="
            )
            approach_res = {k: {} for k in approaches.keys()}
            for approac_name, approach in approaches.items():
                print(
                    f"======================================={approac_name}-{domain}======================================="
                )
                for i in [2, 4, 8, 16, 32, 64, 128]:
                    approach_res[approac_name][i] = approach(
                        model_id,
                        dataset,
                        i,
                        4,
                        5,
                        True,
                        n_classes=len(classes),
                        domain=domain,
                        run=run,
                    )
                approach_res[approac_name]["full"] = approach(
                    model_id,
                    dataset,
                    1000,
                    4,
                    5,
                    True,
                    n_classes=len(classes),
                    domain=domain,
                    run=run,
                    full_train=True,
                )
                runs[run] = approach_res
        domain_res[domain] = runs
        print(
            "=======================================Done======================================="
        )

        with open("reklam24-5.json", "w") as f:
            json.dump(domain_res, f)


def MASSIVE():
    global DATASET
    global ID2LABEL
    global DOMAIN
    domains = [
        "weather",
        "music",
        "cooking",
        "lists",
        "general",
        "email",
        "datetime",
        "play",
        "transport",
        "social",
        "calendar",
        "news",
        "iot",
        "audio",
        "qa",
        "recommendation",
        "takeaway",
        "alarm",
        "full",
    ]
    domain_res = {k: {} for k in domains}
    n_runs = 1
    approaches = {
        "setfit": ex_setfit,
        "transformers": ex_transformers,
    }
    DATASET = "massive"

    for dms in tqdm(domains):
        DOMAIN = dms
        print(
            f"=======================================Domain:{dms}======================================="
        )
        if dms == "full":
            dataset = load_dataset(f"SetFit/amazon_massive_intent_de-DE")
            ID2LABEL = {
                k: v
                for k, v in zip(
                    dataset["train"]["label"], dataset["train"]["label_text"]
                )
            }

        else:
            dataset = load_dataset(f"fathyshalab/massive_{dms}-de-DE")
            ID2LABEL = {
                k: v
                for k, v in zip(dataset["test"]["label"], dataset["test"]["label_name"])
            }

        # reset the labels to be starting with zero and no missing label
        model_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        domain = dms
        classes = np.unique(dataset["train"]["label"])
        print("classes", classes)
        if len(classes) == 1:
            continue
        runs = {r: {} for r in range(n_runs)}
        for run in range(n_runs):
            print(
                f"=======================================Run:{run}======================================="
            )
            approach_res = {k: {} for k in approaches.keys()}
            for approac_name, approach in approaches.items():
                print(
                    f"======================================={approac_name}-{domain}======================================="
                )
                for i in [2, 4, 8, 16, 32, 64, 128]:
                    approach_res[approac_name][i] = approach(
                        model_id,
                        dataset,
                        i,
                        4,
                        5,
                        False,
                        n_classes=len(classes),
                        domain=domain,
                        run=run,
                    )
                approach_res[approac_name]["full"] = approach(
                    model_id,
                    dataset,
                    1000,
                    4,
                    5,
                    False,
                    n_classes=len(classes),
                    domain=domain,
                    run=run,
                    full_train=True,
                )
                runs[run] = approach_res
        domain_res[domain] = runs
        print(
            "=======================================Done======================================="
        )

    with open("massive-5.json", "w") as f:
        json.dump(domain_res, f)


def main():
    print(
        "=======================================reklam======================================="
    )
    reklam24()
    print("#####################Massive#############################################")
    MASSIVE()
    print(
        "=======================================clinic======================================="
    )
    clinic()


if __name__ == "__main__":
    main()
