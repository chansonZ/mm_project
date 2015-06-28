import luigi
import sciluigi as sl
import time
import random
import subprocess as sub

# ====================================================================================================

class CrossValRawData(sl.ExternalTask):

    file_name = luigi.Parameter()

    def out_dataset(self):
        return sl.TargetInfo(self, 'data/{f}'.format(f=self.file_name))

# ====================================================================================================

class CreateFolds(sl.Task):

    # TASK PARAMETERS
    folds_count = luigi.IntParameter()
    fold_index = luigi.IntParameter()
    seed = luigi.Parameter()

    # TARGETS
    in_dataset = None
    def out_traindata(self):
        return sl.TargetInfo(self, self.in_dataset().path + '.fold{0:02}_train'.format(self.fold_index))

    def out_testdata(self):
        return sl.TargetInfo(self, self.in_dataset().path + '.fold{0:02}_test'.format(self.fold_index))

    # CONVENIENCE METHODS
    def count_lines(self, filename):
        out = sub.check_output('wc -l %s' % filename, shell=True)
        return int(out.split(' ')[0])

    def remove_dict_key(self, orig_dict, key):
        new_dict = dict(orig_dict)
        del new_dict[key]
        return new_dict

    def pick_lines(self, dataset, line_nos):
        return [line for i, line in enumerate(dataset) if i in line_nos]

    def run(self):
        linecnt = self.count_lines(self.in_dataset().path)
        line_nos = [i for i in xrange(1, linecnt)]
        random.shuffle(line_nos, lambda: self.seed)

        splits_as_linenos = {}

        set_size = len(line_nos) // int(self.folds_count)

        # Split into splits, in terms of line numbers
        for i in xrange(int(self.folds_count)):
            splits_as_linenos[i] = line_nos[i * set_size : (i+1) * set_size]

        # Write test file
        test_linenos = splits_as_linenos[self.fold_index]
        with self.in_dataset().open() as infile, self.out_testdata().open('w') as testfile:
            for lineno, line in enumerate(infile):
                if lineno in test_linenos:
                    testfile.write(line)

        # Write train file
        train_splits_linenos = self.remove_dict_key(splits_as_linenos, self.fold_index)
        train_linenos = []
        for k, v in train_splits_linenos.iteritems():
            train_linenos.extend(v)
        with self.in_dataset().open() as infile, self.out_traindata().open('w') as trainfile:
            for lineno, line in enumerate(infile):
                if lineno in train_linenos:
                    trainfile.write(line)


# ====================================================================================================

class MergeTrainFolds(sl.Task):

    def run(self):
        time.sleep(0.1)

# ====================================================================================================

class MockTrain(sl.Task):

    in_traindata = None

    def run(self):
        time.sleep(0.1)

# ====================================================================================================

class MockPredict(sl.Task):
    def run(self):
        time.sleep(0.1)

# ====================================================================================================

class MockAssessCrossVal(sl.Task):
    def run(self):
        time.sleep(0.1)
