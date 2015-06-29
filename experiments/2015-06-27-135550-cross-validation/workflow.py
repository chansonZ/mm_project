from crossval import *
import luigi
import sciluigi as sl
import time

# ====================================================================================================
#  New components for Cross-validation - May 8, 2015
# ====================================================================================================

class CrossValidate(sl.WorkflowTask):
    '''
    For now, a sketch on how to implement Cross-Validation as a sub-workflow components
    '''

    # PARAMETERS
    folds_count = luigi.IntParameter()

    #def complete(self):
    #    return False

    def workflow(self):
        # Hard-code this for now ...
        rawdata = sl.new_task(CrossValRawData, 'rawdata', self, file_name='crossval_rawdata.txt')

        # Branch the workflow into one branch per fold
        fold_tasks = {}
        for fold_idx in xrange(self.folds_count):

            create_folds = sl.new_task(CreateFolds, 'create_fold_%d' % fold_idx, self,
                    fold_index = fold_idx,
                    folds_count = self.folds_count,
                    seed = 0.637)

            create_folds.in_dataset = rawdata.out_dataset

            # Plugging in the 'generic' train components, for SVM/LibLinear, here
            #train = sl.new_task(MockTrain, 'train_svm', self)
            #train.in_traindata = create_folds.out_traindata

            #train = MockTrain()

            ## Plugging in the 'generic' predict components, for SVM/LibLinear, here
            #predict = sl.new_task(MockPredict, 'predict', self)
            #predict.in_testdata = create_folds.out_testdata
            #predict.in_svmmodel = train.out_svmmodel

            fold_tasks[fold_idx] = {}
            fold_tasks[fold_idx]['create_folds'] = create_folds
            #fold_tasks[fold_idx]['train'] = train
            #fold_tasks[fold_idx]['predict'] = predict

        # Collect the prediction targets from the branches above, into one dict, to feed
        # into the specialized assess component below
        #predict_targets = { i : fold_tasks[fold_idx]['predict'] for i in xrange(self.folds_count) }
        #assess = sl.new_task(MockAssessCrossVal, 'assess', self, folds_count=self.folds_count)
        #assess.in_predict_targets = predict_targets

        return_tasks = [fold_tasks[fold_idx]['create_folds'] for fold_idx in xrange(self.folds_count)]
        return return_tasks

# ====================================================================================================

if __name__ == '__main__':
    sl.run_locally()
