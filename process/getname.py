import os
import pandas as pd


def getname(args):
    names = [n.split('.h5')[0] for n in os.listdir(f'{args.save_dir}/patches') if n.endswith('.h5')]

    dataframe = pd.DataFrame({'slide_id': names})
    dataframe.to_csv(f"{args.save_dir}/process_list.csv", sep=',')
