import luigi
import sciluigi as sl

# ====================================================================================================

class CrossValRawData(sl.ExternalTask):
    def out_dataset(self):
        return sl.TargetInfo(self, 'data/cv_rawdata.txt')

# ====================================================================================================

class CrossValSplitByLines(sl.Task):

    # INPUT TARGETS
    in_dataset = None

    # TASK PARAMETERS
    folds_count = luigi.Parameter()

    def output(self):
        return { 'split_%d' % i: luigi.LocalTarget(self.get_input('dataset_target').path + '.split_%d' % i) for i in xrange(int(self.splits_count)) }

    # CONVENIENCE METHODS
    def count_lines(self, filename):
        status, output = self.lx(['cat', filename, '|', 'wc -l'])
        return int(output)

    def remove_dict_key(self, orig_dict, key):
        new_dict = dict(orig_dict)
        del new_dict[key]
        return new_dict

    def pick_lines(self, dataset, line_nos):
        return [line for i, line in enumerate(dataset) if i in line_nos]

    def run(self):
        linecnt = self.count_lines(self.get_input('dataset_target').path)
        line_nos = [i for i in xrange(1, linecnt)]
        random.shuffle(line_nos)

        splits_as_linenos = {}

        set_size = len(line_nos) // int(self.folds_count)

        # Split into splits, in terms of line numbers
        for i in xrange(int(self.folds_count)):
            splits_as_linenos[i] = line_nos[i * set_size : (i+1) * set_size]

        for i, split_id in enumerate(self.output()):
            with self.get_input('dataset_target').open() as infile, self.output()[split_id].open('w') as outfile:
                lines = self.pick_lines(infile, splits_as_linenos[i])
                outfile.writelines(lines)

# ====================================================================================================

class MergeTrainFolds(sl.Task):

    def run(self):
        time.sleep(1)

# ====================================================================================================

class MockTrain(sl.Task):
    def run(self):
        time.sleep(1)

# ====================================================================================================

class MockPredict(sl.Task):
    def run(self):
        time.sleep(1)

# ====================================================================================================

class MockAssessCrossVal(sl.Task):
    def run(self):
        time.sleep(1)

# ====================================================================================================

if __name__ == '__main__':
    sl.run_locally()
