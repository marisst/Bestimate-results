import os
import json

RESULTS_FOLDER_NAME = "results"
CORRECT_COUNTS = {
    "all_1_1": (54642, 1435732),
    "all_1_10": (44499, 1289006),
    "all_1_20": (35966, 1106787),
    "all_200_1": (41686, 374039),
    "all_200_10": (32598, 313850),
    "all_200_20": (25145, 237287),
    "all_500_1": (32754, 250599),
    "all_500_10": (24585, 179790),
    "all_500_20": (19992, 158405),
    "all_1000_1": (23768, 145730),
    "all_1000_10": (16945, 118635),
    "all_1000_20": (9686, 52840),

}

def subdirs(root_dir):
    for item in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, item)):
            yield item


def sanity_check(architecture, dataset):

    folder_name = os.path.join(RESULTS_FOLDER_NAME, architecture, dataset)

    results_filename = os.path.join(folder_name, "results.txt")
    with open(results_filename) as results_file:
        for i, line in enumerate(results_file):
            if i == 1:
                words = line.split()

                labeled_count = int(words[0])
                if labeled_count != CORRECT_COUNTS[dataset][0]:
                    print("Incorrect labeled datapoint count: %s, %s" % (dataset, architecture))
                    return False

                unlabeled_count = int(words[3])
                if unlabeled_count != CORRECT_COUNTS[dataset][1]:
                    print("Incorrect unlabeled datapoint count: %s, %s" % (dataset, architecture))
                    return False

    for run in subdirs(folder_name):
        notes_filename = os.path.join(folder_name, run, "notes.txt")
        with open(notes_filename) as notes_file:
            lines = notes_file.readlines()
            last_configuration_line = lines.index("}\n")
            configuration_string = "".join(lines[:last_configuration_line+1])
            configuration = json.loads(configuration_string)

            if configuration["training_dataset_id"] != dataset.split("_")[0]:
                print("Wrong dataset, %s architecture, %s dataset, %s run" % (architecture, dataset, run))
                return False

            if configuration["min_project_size"] != int(dataset.split("_")[1]):
                print("Wrong min_project_size, %s architecture, %s dataset, %s run" % (architecture, dataset, run))
                return False

            if configuration["min_word_count"] != int(dataset.split("_")[2]):
                print("Wrong min_word_count, %s architecture, %s dataset, %s run" % (architecture, dataset, run))
                return False

            if configuration["word_embeddings"]["type"] != architecture.split("-")[0]:
                print("Wrong embeddings, %s architecture, %s dataset, %s run" % (architecture, dataset, run))
                return False

            if (configuration["model_params"].get("lstm_count") == None and architecture.split("-")[1] == "1"
                or architecture.split("-")[1].isdigit() and configuration["model_params"]["lstm_count"] == int(architecture.split("-")[1])
                or configuration["model_params"]["lstm_count"] == 3 and architecture.split("-")[1] == "bi") == False:
                print(architecture.split("-")[1], configuration["model_params"].get("lstm_count"))
                print("Wrong context network type, %s architecture, %s dataset, %s run" % (architecture, dataset, run))
                return False

            if (configuration["model_params"].get("conform_type") == None and architecture.split("-")[2] == "hway"
                or configuration["model_params"].get("conform_type") == architecture.split("-")[2]) == False:
                print("Wrong context transformation network type, %s architecture, %s dataset, %s run" % (architecture, dataset, run))
                return False


    
    return True


not_passed = []
for architecture in subdirs(RESULTS_FOLDER_NAME):
    for dataset in subdirs(os.path.join(RESULTS_FOLDER_NAME, architecture)):
        if sanity_check(architecture, dataset) == False:
            not_passed.append((architecture, dataset))

if len(not_passed) > 0:
    print("%d architecture tests didn't pass the sanity check" % len(not_passed))
else:
    print("All architectures and datasets passed sanity check")