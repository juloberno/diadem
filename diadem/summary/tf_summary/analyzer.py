# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


from diadem.summary.tf_summary.reader import Reader
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





class Analyzer:
    def __init__(self, scalar_size):
        self.sum_reader = Reader(scalar_size)
        self.axes = None
        self.line_styles = ['-','-.','--',':',(0,(3,1)),(0,(10,5)),(0,(20,5))]
        self.line_styles += self.line_styles
        self.line_styles += self.line_styles
        self.marker_styles = ['o', 'v', '^', '<',  '>' ,  's'  ,'p'  ,'*'  ,'h' ,'H' ,'+'  ,'x'  ,'D' ,'d']
        self.marker_styles += self.marker_styles
        self.marker_styles += self.marker_styles
        self.markevery = 0.01
        self.x_of_ylabel=-0.2

    def add_logs(self, **kwargs):
        return self.sum_reader.add_logs(kwargs)
    
    
    def load_data_frame_from_event_file(self, **kwargs):
        
        reload = False
        df = None
        matchings = None
        if "reload" in kwargs and kwargs["reload"]:
            reload = True
        if reload:
            if "tag_list" in kwargs:
                tag_list = kwargs["tag_list"]
                del kwargs["tag_list"]
            else:
                print("error: no tag_list given")
                return
            
            if not self.sum_reader.number_of_log_files > 1:
                self.sum_reader.add_logs(**kwargs)
                if not self.sum_reader.number_of_log_files > 1:
                    print("error: could find logfiles")
                    return
            
            df, matchings = self.sum_reader.fetch_data_frame(tag_list, **kwargs)
            if "data_path" in kwargs:
                if not os.path.exists(os.path.dirname(kwargs["data_path"])):
                    os.makedirs(os.path.dirname(kwargs["data_path"]))
                df.to_pickle(kwargs["data_path"])
            
        elif "data_path" in kwargs:
            df = pd.read_pickle(kwargs["data_path"])
            if "tag_list" in kwargs:
                tag_list = kwargs["tag_list"]
                cols = list(df.columns.values)
                matchings = []
                for tag in tag_list:
                    matchings.append( [col for col in cols if tag in col][0])
            else:
                print("error: no matchings given")
            
        else:
            print("error: no data path given.")
            return
            

        return df, matchings

    def plot_over_training_steps(self, df, matchings, **kwargs):

        figsize = [6, 5]
        if "figsize" in kwargs:
            figsize = kwargs["figsize"]

        axes = []

        if "subplots" in kwargs:
            if kwargs["subplots"]:
                fig, ax = plt.subplots(len(matchings), figsize=figsize, sharex=True)
                if len(ax) == 1:
                    axes.append(ax)
                else:
                    axes = ax
            else:
                self.fig, self.axes = plt.figure(num=len(matchings), figsize=figsize)
        elif "axes" in kwargs:
            axes = kwargs["axes"]
            fig = axes.get_figure()
            if not isinstance(axes, list):
                axes = [axes]
                fig = [fig]
        else:
            fig = []
            axes = []
            for i in range(0,len(matchings)):
                fig.append( plt.figure(num=i, figsize=figsize))
                axes.append( self.fig[i].gca() )

        

       # sns.set()
        
        if not isinstance(matchings, list):
            matchings = [matchings]

        x_column = kwargs.get('x_column', 'steps')

        idx = 0
        for match in matchings:
            data_frame = df.copy()
            next_axes = axes[idx]

            if "normalize_x" in kwargs:
                data_frame[x_column] = data_frame[x_column]/ kwargs["normalize_x"]

            y_norm = 1
            if "normalize_y" in kwargs and isinstance(kwargs["normalize_y"], list):
                y_norm = kwargs["normalize_y"][idx]
            elif "normalize_y" in kwargs:
                y_norm = kwargs["normalize_y"]

            data_frame[match] = data_frame[match] / y_norm
            
            if "y_range" in kwargs:
                if not isinstance(kwargs["y_range"][0], list):
                    y_range = kwargs["y_range"]
                else:
                    y_range = np.array(kwargs["y_range"][idx]).astype(float)
                data_frame = data_frame[(data_frame[match] < y_range[1]) & (data_frame[match] > y_range[0])]
            
            data_frame = data_frame.dropna()
            if "rolling_mean" in kwargs:
                data_frame[match] = data_frame[match].rolling(kwargs['rolling_mean']).mean()

            next_axes = sns.lineplot(data=data_frame, x=x_column, y=match, hue="experiment", units="cycle", ax=next_axes, estimator=None, ci="sd")

           # if "x_range" in kwargs:
            #      x_range = np.array(kwargs["x_range"])
           #       next_axes.set_xlim((x_range[0], x_range[1]))


            if "y_range" in kwargs:
                if not isinstance(kwargs["y_range"][0], list):
                    y_range = kwargs["y_range"]
                else:
                    y_range = np.array(kwargs["y_range"][idx]).astype(float)
                next_axes.set_ylim((y_range[0], y_range[1]))
                
                
            if "x_range" in kwargs:
                if not isinstance(kwargs["x_range"][0], list):
                    x_range = kwargs["x_range"]
                else:
                    x_range = np.array(kwargs["x_range"][idx]).astype(float)
                next_axes.set_xlim((x_range[0], x_range[1]))
                
            st_idx = 0
            ls =  self.line_styles
            ms = self.marker_styles
            for line in next_axes.lines:
                if "linestyles" in kwargs:
                    if kwargs["linestyles"]:
                         line.set_linestyle(ls[st_idx])
                if "markerstyles" in kwargs:
                    if kwargs["markerstyles"]:
                         line.set_marker(ms[st_idx])
                         line.set(markevery=self.markevery)
                st_idx += 1

            next_axes.legend()

            next_axes.get_figure().show()

            idx+=1

        if "x_label" in kwargs:
            for ax in axes:
                ax.set_xlabel(kwargs["x_label"])

        # same position of ylabels
        #for ax in axes:
          #  ax.yaxis.set_label_coords(self.x_of_ylabel,0.5)


        if "y_label" in kwargs:
            t = 0
            for ax in axes:
                if isinstance(kwargs["y_label"], list):
                    if len(kwargs["y_label"]):
                        ax.set_ylabel(kwargs["y_label"][t])
                else:
                    ax.set_ylabel(kwargs["y_label"])

                t +=1

        if "lg_labels" in kwargs:

            for ax in axes:
                leg = ax.get_legend()
                for t in range(0, len(leg.texts)):
                    if len(kwargs["lg_labels"])-1 < t:
                        continue
                    else:
                        leg.texts[t].set_text(kwargs["lg_labels"][t])
                    t += 1
                ax.legend()

                plt.gcf().canvas.draw_idle()

        return axes

    def add_experiment_folder(self, exp_dir, exp_list=[], **kwargs):
        """

        :param exp_dir: Directory which contains multiple experiments in subfolder, where each experiment has its logfiles in the same folder
        :param exp_list: List of subfolder to consider in experiment
        :return: number of succesfully added logfiles or 0
        """
        if len(exp_list) == 0:
            exp_list = next(os.walk(exp_dir))[1]

        for exp in exp_list:
            subdirs = [x[0] for x in os.walk(exp_dir)]
            for subdir in subdirs:
                if exp in subdir:
                    log_dir = os.path.join(subdir.split(exp)[0],exp)
                    self.sum_reader.add_logs(logdir=log_dir,name=exp, **kwargs)
                    break
        return self.sum_reader.number_of_log_files
    
    def get_latest_model_files(self, exp_name,cycle_num=1):
        log_file_names = self.sum_reader.log_files
        
        ckpt_file = None
        model_file = None
        param_file = None
        
        if not len(log_file_names) > 1:
            print("error: no logfiles added yet.")
            
        for log_file in log_file_names:
            if exp_name == log_file[0] and cycle_num == int(log_file[1]):
                paths = log_file[2].split("cycle_")
                model_folder = os.path.join(paths[0],"cycle_" + str(cycle_num),"model")
                
                latest_file = self.get_latest_file(model_folder, filter="cycle_" + str(cycle_num))
                base_name = latest_file.split(".meta")[0]
                
                model_file = os.path.join(model_folder,base_name+".meta")
                ckpt_file = os.path.join(model_folder, base_name)
                param_file = os.path.join(paths[0], "parameters.json")
                
                break
        return model_file, ckpt_file, param_file
                
    def get_latest_file(self, path, filter=None):
        gt = os.path.getmtime  # change if you want something else
        if filter is not None:
            filtered_files = [f for f in os.listdir(path) if filter in f]
        else:
            filtered_files = os.listdir(path)
            
        newest = max([(f, gt(os.path.join(path, f))) for f in filtered_files])[0]
    
        return newest
                
            
        


if __name__ == "__main__":
    analyzer = Analyzer(0)

    print(analyzer.add_logs(logdir="/home/julo/jb_git/Programming/Python/DDQ_line_new/ddqn_per_binary_reward/tmp_results/log"))

    data = analyzer.fetch_data('average_reward', exact_match=False)