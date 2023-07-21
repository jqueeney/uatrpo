"""Entry point for visualizing results."""
import numpy as np

from uatrpo.common.plot_utils import create_plotparser, plot_setup, create_plot

def main():
    """Parses inputs, creates and saves plot."""
    parser = create_plotparser()
    args = parser.parse_args()

    x = np.arange(0,args.timesteps+1,args.interval)
    results_list = plot_setup(args.import_path,args.import_files,
        x,args.window,args.metric)

    create_plot(x,results_list,
        args.se_val,args.labels,args.figsize,args.save_path,args.save_name)


if __name__=='__main__':
    main()