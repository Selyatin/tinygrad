import sklearn.datasets as sd
import argparse

p = argparse.ArgumentParser(description="Create a toy dataset for testing tinygrad.")
p.add_argument("--task", type=str, choices=["classification", "regression"], dest="task", required=True)
p.add_argument("--rows", type=int, default=100, dest="rows", required=False, help="Number of data points")
p.add_argument("--features", type=int, default=4, dest="features", required=False, help="Number of features")
p.add_argument("--file", type=str, default=100, dest="file", required=True, help="Filename for dataset")
o = p.parse_args()

if o.task == "classification":
    with open(o.file, "w") as fh:
        x, y = sd.make_classification(n_samples=o.rows, n_features=o.features, n_informative=o.features, n_redundant=0)
        for x_row, y_row in zip(x, y):
            fh.write(("{:.4f},"*x_row.size).format(*x_row))
            fh.write("{}\n".format(float(y_row)))
