import numpy as np
import pandas as pd
from absl import app
from absl import flags
from nbc import clustering

FLAGS = flags.FLAGS
flags.DEFINE_integer("k", default=5, help="Nearest neighbours count.")
flags.DEFINE_string("path", default=None, short_name="p", help="Path to dataset as csv.")
flags.DEFINE_boolean("use_ti", default=False, short_name="ti",
                     help="Whether to use NBC with a Triangle Inequality (TI)")
flags.DEFINE_list("reference_point", default=[], short_name="rp",
                  help="Reference point if using TI - by default list of minimums.")
flags.mark_flag_as_required("path")


def run(_):
    df = pd.read_csv(FLAGS.path, header=None, delimiter=",")
    points = df.astype(np.float64)
    if FLAGS.use_ti:
        reference_point = (np.array(FLAGS.reference_point) if FLAGS.reference_point
                           else np.array([min(points[0]), min(points[1])]))
    else:
        reference_point = None
    clusters = clustering.nbc(points.values, FLAGS.k, reference_point=reference_point)
    # TODO - add saving to file  instead of printing
    print(clusters)


def main():
    app.run(run)


if __name__ == "__main__":
    main()
