import os
import matplotlib.pyplot as plt
import numpy

RESULTS_FOLDER_NAME = "results"
GRAPHS_FOLDER_NAME = "graphs"
DATASETS = {"all_1_1", "all_1_10", "all_1_20", "all_200_1", "all_200_10", "all_200_20", "all_500_1", "all_500_10", "all_500_20", "all_1000_1", "all_1000_10", "all_1000_20"}
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
    "all_1000_20": (9686, 52840)}

def subdirs(root_dir):
    for item in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, item)):
            yield item


def get_result(architecture, dataset):

    results_filename = os.path.join(RESULTS_FOLDER_NAME, architecture, dataset, "results.txt")
    with open(results_filename) as results_file:
        lines = results_file.readlines()[2:]
        val_results = numpy.array([float(line[:-1].split()[-1]) for line in lines])
        result = 1 - min(val_results)
        confidence = min(val_results) / numpy.average(val_results)
    return result * 100, confidence * 100


if not os.path.exists(GRAPHS_FOLDER_NAME):
    os.mkdir(GRAPHS_FOLDER_NAME)

architecture_result = [{}, {}]
for dataset in sorted(DATASETS):
    results = []
    for architecture in sorted(subdirs(RESULTS_FOLDER_NAME)):
        if os.path.exists(os.path.join(RESULTS_FOLDER_NAME, architecture, dataset)):
            result = get_result(architecture, dataset)
            results.append((architecture, result))
            if architecture_result[0].get(architecture) == None:
                architecture_result[0][architecture] = [result[0]]
                architecture_result[1][architecture] = [result[1]]
            else:
                architecture_result[0][architecture].append(result[0])
                architecture_result[1][architecture].append(result[1])

    print(dataset)
    print("%d labeled datapoints" % CORRECT_COUNTS[dataset][0])
    print("%0.2f%% average result" % numpy.average([r[1][0] for r in results]))
    for result in results:
        print("%s architecture - %0.2f%% better than baseline with %0.0f%% confidence" % (result[0], result[1][0], result[1][1]))
    print()

    plt.figure(figsize=(2, 1))
    axs = plt.gca()
    axs.clear()
    
    barlist = axs.bar(numpy.arange(len(results)), [r[1][0] for r in results])
    
    colors = ["coral", "orange", "yellowgreen", "mediumaquamarine", "dodgerblue"]
    for i, bar in enumerate(barlist):
        bar.set_color(colors[i])
    
    axs.get_xaxis().set_visible(False)
    axs.set_ylim(bottom=0, top=10.5)
    axs.set_yticks(numpy.arange(1, 11, step = 2))

    if dataset.split("_")[2] != "1":
        for tic in axs.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False

    axs.grid(True, color="dimgray")
    plt.savefig(os.path.join(GRAPHS_FOLDER_NAME, dataset + "-results.png"), bbox_inches='tight', pad_inches = 0, transparent=True)

print("Architecture results")
for architecture, result in architecture_result[0].items():
    print("%s architecture - %0.2f%% average baseline outperforming" % (architecture, numpy.average(result)))
print()

print("Architecture confidence")
for architecture, result in architecture_result[1].items():
    print("%s architecture - %0.2f%% average confidence" % (architecture, numpy.average(result)))
print()