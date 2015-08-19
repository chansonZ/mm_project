from crossval import *
from mmcomp import *
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
    replicate_id = luigi.Parameter()

    #def complete(self):
    #    return False

    def workflow(self):
        # Existing smiles data
        mmtestdata = self.new_task('mmtestdata', ExistingSmiles, dataset_name='mm_test', replicate_id=self.replicate_id)

        createrepl = self.new_task('createrepl', CreateReplicateCopy, replicate_id=self.replicate_id)
        createrepl.in_file = mmtestdata.out_smiles


        # Hard-code this for now ...
        # rawdata = self.new_task('rawdata', CrossValRawData, file_name='raw/mm_test.smiles')

        # Branch the workflow into one branch per fold
        fold_tasks = {}
        for fold_idx in xrange(self.folds_count):

            # Task: create_folds
            create_folds = self.new_task('create_fold_%d' % fold_idx, CreateFolds,
                    fold_index = fold_idx,
                    folds_count = self.folds_count,
                    seed = 0.637)

            create_folds.in_dataset = createrepl.out_copy

            #TODO:
            # - Add "existing data" task for mm_test.smiles
            # - Convert existing MM tasks to new API?
            # - Replicate the preprocessing chain from the previous MM

            # Task names
            # - ExistingSmiles
            # - GenerateSignaturesFilterSubstances
            # - GenerateUniqueSignaturesCopy
            # - SampleTrainAndTest
            # - CreateSparseTrainDataset
            # - CreateSparseTestDataset
            # - TrainLinearModel
            # - PredictLinearModel
            # - AssessLinearRegression
            # - CreateReport


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
