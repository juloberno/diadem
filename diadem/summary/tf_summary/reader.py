# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import numpy as np
import os
import pandas as pd
import re

# Import the event accumulator from Tensorboard. Location varies between Tensorflow versions. Try each known location until one works.
eventAccumulatorImported = False
# TF version < 1.1.0
if (not eventAccumulatorImported):
    try:
        from tensorflow.python.summary import event_accumulator
        eventAccumulatorImported = True
    except ImportError:
        eventAccumulatorImported = False
# TF version = 1.1.0
if (not eventAccumulatorImported):
    try:
        from tensorflow.tensorboard.backend.event_processing import event_accumulator
        eventAccumulatorImported = True
    except ImportError:
        eventAccumulatorImported = False
# TF version >= 1.3.0
if (not eventAccumulatorImported):
    try:
        from tensorboard.backend.event_processing import event_accumulator
        eventAccumulatorImported = True
    except ImportError:
        eventAccumulatorImported = False
# TF version = Unknown
if (not eventAccumulatorImported):
    raise ImportError(
        'Could not locate and import Tensorflow event accumulator.')


class Reader:
    def __init__(self, scalar_size):
        test = 0
        self.event_accs = []  # a list of event accumulators
        # save logfile names as these cannot be retrieved from event_acc
        self.log_file_names = []

        # specify how much data to load
        self.tf_size_guidance = {'compressedHistograms': 0,
                                 'images': 0, 'scalars': scalar_size, 'histograms': 0}
        self.loaded = False

    @property
    def number_of_log_files(self):
        return len(self.log_file_names)

    @property
    def log_files(self):
        return self.log_file_names

    def add_logs(self, **kwargs):
        if 'logfile' in kwargs:
            if 'name' in kwargs:
                log_name = str(kwargs['name'])
            else:
                log_name = kwargs['logfile']
            event_acc = event_accumulator.EventAccumulator(
                kwargs['logfile'], self.tf_size_guidance)
            event_acc.Reload()
            self.event_accs.append(event_acc)
            self.log_file_names.append(
                (kwargs["logfile"], cycle_num, kwargs["logfile"]))

        elif 'logdir' in kwargs:
            idx = -1

            for root, dirs, files in os.walk(kwargs['logdir']):
                for file in files:
                    if "events.out" in file:
                        idx += 1

                        filename = os.path.join(root, file)
                        cycle_num = 0
                        if 'name' in kwargs:
                            if "cycles_named" in kwargs:
                                cycles_named = kwargs["cycles_named"]
                            else:
                                cycles_named = False
                            if cycles_named:
                                try:
                                    found = re.search("cycle_\d",filename)
                                    cycle_num = str(found[0].rstrip()[-1])
                                    log_name = kwargs['name']
                                except:
                                    raise ValueError("Logfile: {} does not contain cycle num".format(filename))
                            else:
                                cycle_num = str(idx)
                                log_name = kwargs['name']
                        else:
                            log_name = filename

                        event_acc = event_accumulator.EventAccumulator(
                            filename, self.tf_size_guidance)
                        self.event_accs.append(event_acc)
                        self.log_file_names.append(
                            (log_name, cycle_num, filename))

            if idx >= 0:
                print("I added " + str(idx+1) +
                      " different logfiles: " + filename)
            else:
                print("No logfiles found")

        tags = []
        for event_acc in self.event_accs:
            tags.append(event_acc.Tags())

        return tags

    def fetch_data_frame(self, tag_list, **kwargs):
        if 'reload' in kwargs:
            reload = kwargs['reload']
        else:
            reload = False

        if 'exact_match' in kwargs:
            exact_match = kwargs['exact_match']
        else:
            exact_match = False

        # reload all data if necessary
        if reload or not self.loaded:
            for event_acc in self.event_accs:
                event_acc.Reload()
            self.loaded = True

        # Get actual data
        fetched_dict = {}
        data_type = None
        log_file_idx = 0

        frames = []
        for event_acc in self.event_accs:
            event_acc_tags = event_acc.Tags()
            match_found = False
            frame = pd.DataFrame()

            matchings = []
            for it_data_type, tags in event_acc_tags.items():

                for tag in tag_list:
                    if isinstance(tags, (list, str)) and len(tags) > 0:
                        matchings_new = [t for t in tags if tag in t]
                    else:
                        continue

                    if len(matchings_new) > 0:
                        # perfect, one match found
                        # check if match is same as previous match

                        for matching in matchings_new:
                            if data_type is None:
                                print("Found first match.")
                            elif not data_type is None and data_type == matching:
                                if exact_match:
                                    print(
                                        "match is not exactly the same as first match. We skip this...")
                                    continue
                                else:
                                    print(
                                        "match is not exactly the same as first match. But we use it...")

                            if it_data_type is "images":
                                print(
                                    "found image data. Images are not yet supported.")
                                return []
                            elif it_data_type is "audio":
                                print(
                                    "found audio data. Audio is not yet supported.")
                                return []
                            elif it_data_type is "histogram":
                                print(
                                    "found histogram data. Histograms are not yet supported.")
                                data = event_acc.Histograms(matching)
                                curr_fetched_data[matching] = np.array(data)
                            elif it_data_type is "scalars":
                                data = event_acc.Scalars(matching)
                                data = np.array(data)
                                normalize_steps = 1
                                if "normalize_steps" in kwargs:
                                    normalize_steps = kwargs["normalize_steps"]

                                frame["steps"] = data[:, 1]/normalize_steps
                                frame[matching] = data[:, 2]

                                if "running_average" in kwargs:
                                    val_mean = frame[matching].rolling(
                                        window=kwargs["running_average"]).mean()
                                    # val_mean = values[match].rolling(window=kwargs["running_average"]).mean()
                                    # https://stackoverflow.com/questions/12957582/matplotlib-plot-yerr-xerr-as-shaded-region-rather-than-error-bars
                                    # to add traces??
                                    frame[matching] = val_mean

                                frame["time"] = data[:, 0] - data[0, 0]
                                frame["experiment"] = self.log_file_names[log_file_idx][0]
                                frame["cycle"] = self.log_file_names[log_file_idx][1]

                        matchings = matchings + matchings_new
                        match_found = True
                    else:
                        print("Error: no matching found for tag: " + tag +
                              " in Logfile: " + self.log_file_names[log_file_idx][2])

            if match_found:
                frames.append(frame)

            log_file_idx += 1

        full_frame = pd.concat(frames, ignore_index=True)

        return full_frame, matchings


if __name__ == "__main__":
    analyzer = Reader()

    print(analyzer.add_logs(
        logdir="/home/julo/jb_git/Programming/Python/DDQ_line_new/ddqn_per_binary_reward/tmp_results/log"))

    data = analyzer.fetch_data('average_reward', exact_match=False)
    test = 0
