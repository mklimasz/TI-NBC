import numpy as np
import pandas as pd
from absl import app
from absl import flags
from nbc import clustering

FLAGS = flags.FLAGS
flags.DEFINE_integer("k", default=5, help="Nearest neighbours count.")
flags.DEFINE_string("path", default=None, short_name="p", help="Path to dataset as comma separated csv.")
flags.DEFINE_boolean("use_ti", default=False, short_name="ti",
                     help="Whether to use NBC with a Triangle Inequality (TI).")
flags.DEFINE_list("reference_point", default=[], short_name="rp",
                  help="Reference point if using TI - by default list of minimums.")
flags.DEFINE_string("output_path", default="clusters.csv", short_name="o", help="Output path for csv with clusters.")
flags.mark_flag_as_required("path")


def run(_):
    df = pd.read_csv(FLAGS.path, header=None, delimiter=",")
    points = df.astype(np.float64)
    if FLAGS.use_ti:
        reference_point = (np.array(FLAGS.reference_point).astype(np.float64) if FLAGS.reference_point
                           else np.array(points.values.min(axis=0)))
    else:
        reference_point = None
    clusters = clustering.nbc(points.values, FLAGS.k, reference_point=reference_point)
    with open(FLAGS.output_path, "w") as csv:
        for _, value in sorted(clusters.items(), key=lambda kv: kv[0]):
            csv.write(f"{value}\n")


def main():
    app.run(run)


if __name__ == "__main__":
    main()
