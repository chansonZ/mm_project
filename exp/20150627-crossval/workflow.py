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
    min_height = luigi.Parameter()
    max_height = luigi.Parameter()

    def workflow(self):
        # Initialize tasks
        mmtestdata = self.new_task('mmtestdata', ExistingSmiles,
                replicate_id=self.replicate_id,
                dataset_name='mm_test_small')
        gensign = self.new_task('gensign', GenerateSignaturesFilterSubstances,
                replicate_id=self.replicate_id,
                min_height = self.min_height,
                max_height = self.max_height,
                slurminfo = sl.SlurmInfo(
                    runmode=sl.RUNMODE_HPC, # For debugging
                    project='b2013262',
                    partition='devcore',
                    cores='2',
                    time='15:00',
                    jobname='MMGenSignTest',
                    threads='2'
                ))
        replcopy = self.new_task('replcopy', CreateReplicateCopy,
                replicate_id=self.replicate_id)

        # Connect tasks
        gensign.in_smiles = mmtestdata.out_smiles
        replcopy.in_file = gensign.out_signatures

        # Branch the workflow into one branch per fold
        fold_tasks = {}
        for fold_idx in xrange(self.folds_count):
            # Init tasks
            create_folds = self.new_task('create_fold_%d' % fold_idx, CreateFolds,
                    fold_index = fold_idx,
                    folds_count = self.folds_count,
                    seed = 0.637)

            # Connect tasks
            create_folds.in_dataset = replcopy.out_copy

            #TODO:
            # - Convert existing MM tasks to new API?
            # - Replicate the preprocessing chain from the previous MM

            # Task names
            # - [x] ExistingSmiles
            # - [x] GenerateSignaturesFilterSubstances
            # - [ ] GenerateUniqueSignaturesCopy
            # - [ ] SampleTrainAndTest
            # - [ ] CreateSparseTrainDataset
            # - [ ] CreateSparseTestDataset
            # - [ ] TrainLinearModel
            # - [ ] PredictLinearModel
            # - [ ] AssessLinearRegression
            # - [ ] CreateReport

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
