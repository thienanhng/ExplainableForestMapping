import pandas as pd

def get_fns(fns_csv_fn, *args):
    """Get filenames from csv file as lists of strings.
    args are column names"""
    
    df = pd.read_csv(fns_csv_fn)
    if len(args) > 1:
        output = []
        for attr in args:
            try:
                output.append(df[attr].values)
            except KeyError:
                output.append(None)
    else:
        output = df[args[0]].values
    return output
