from mmcomp import *
import luigi
import sciluigi as sl
import time

# ================================================================================

class MMLinear(sl.WorkflowTask):
    '''
    This class runs the MM Workflow using LibLinear
    as the method for doing machine learning
    '''

    # WORKFLOW PARAMETERS
    dataset_name = luigi.Parameter(default='mm_test_small')
    replicate_id = luigi.Parameter()
    test_size = luigi.Parameter()
    train_size = luigi.Parameter()
    sampling_seed = luigi.Parameter(default=None)
    sampling_method = luigi.Parameter()
    lin_type = luigi.Parameter()
    lin_cost = luigi.Parameter()
    slurm_project = luigi.Parameter()
    parallel_lin_train = luigi.BooleanParameter()
    #folds_count = luigi.Parameter()

    def workflow(self):
        '''
        The dependency graph is defined here!
        '''
        # --------------------------------------------------------------------------------
        existing_smiles = self.new_task('existing_smiles', ExistingSmiles,
                dataset_name = self.dataset_name,
                replicate_id = self.replicate_id)
        # --------------------------------------------------------------------------------
        gen_sign_filter_subst = self.new_task('gen_sign_filter_subst', GenerateSignaturesFilterSubstances,
                min_height = 1,
                max_height = 3,
                dataset_name = self.dataset_name,
                replicate_id = self.replicate_id,
                slurminfo = sl.SlurmInfo(
                    runmode=sl.RUNMODE_HPC, # For debugging
                    project=self.slurm_project,
                    partition='devcore',
                    cores='2',
                    time='15:00',
                    jobname='MMSampleTrainTest',
                    threads='2'
                ))
        gen_sign_filter_subst.in_smiles = existing_smiles.out_smiles
        # --------------------------------------------------------------------------------
        create_unique_sign_copy = self.new_task('create_unique_sign_copy', CreateReplicateCopy,
                replicate_id = self.replicate_id)
        create_unique_sign_copy.in_file = gen_sign_filter_subst.out_signatures
        # ------------------------------------------------------------------------
        # RANDOM TRAIN/TEST SAMPLING
        # ------------------------------------------------------------------------
        if self.sampling_method == 'random':

            sample_train_and_test = self.new_task('sample_train_and_test', SampleTrainAndTest,
                    seed = self.sampling_seed,
                    test_size = self.test_size,
                    train_size = self.train_size,
                    sampling_method = self.sampling_method,
                    dataset_name = self.dataset_name,
                    replicate_id = self.replicate_id,
                    slurminfo = sl.SlurmInfo(
                        runmode=sl.RUNMODE_HPC, # For debugging
                        project=self.slurm_project,
                        partition='devcore',
                        cores='2',
                        time='15:00',
                        jobname='MMSampleTrainTest',
                        threads='2'
                    ))
            sample_train_and_test.in_signatures = create_unique_sign_copy.out_copy
        # ------------------------------------------------------------------------
        # BCUT TRAIN/TEST SAMPLING
        # ------------------------------------------------------------------------
        elif self.sampling_method == 'bcut':
            # ------------------------------------------------------------------------
            bcut_preprocess = self.new_task('bcut_preprocess', BCutPreprocess,
                    replicate_id = self.replicate_id,
                    dataset_name = self.dataset_name,
                    slurminfo = sl.SlurmInfo(
                        runmode=sl.RUNMODE_HPC, # For debugging
                        project=self.slurm_project,
                        partition='devcore',
                        cores='2',
                        time='15:00',
                        jobname='MMSampleTrainTest',
                        threads='2'
                    ))
            bcut_preprocess.in_signatures = create_unique_sign_copy.out_signatures
            # ------------------------------------------------------------------------
            sample_train_and_test = self.new_task('sample_train_and_test', BCutSplitTrainTest,
                    train_size = self.train_size,
                    test_size = self.test_size,
                    replicate_id = self.replicate_id,
                    dataset_name = self.dataset_name,
                    slurminfo = sl.SlurmInfo(
                        runmode=sl.RUNMODE_HPC, # For debugging
                        project=self.slurm_project,
                        partition='devcore',
                        cores='2',
                        time='15:00',
                        jobname='MMSampleTrainTest',
                        threads='2'
                    ))
            sample_train_and_test.in_bcut_preprocessed = bcut_preprocess.out_bcut_preprocessed
        # (end if)
        # ------------------------------------------------------------------------
        create_sparse_train_dataset = self.new_task('create_sparse_train_dataset', CreateSparseTrainDataset,
                dataset_name = self.dataset_name,
                replicate_id = self.replicate_id,
                slurminfo = sl.SlurmInfo(
                    runmode=sl.RUNMODE_HPC, # For debugging
                    project=self.slurm_project,
                    partition='devcore',
                    cores='2',
                    time='15:00',
                    jobname='MMSampleTrainTest',
                    threads='2'
                ))
        create_sparse_train_dataset.in_traindata = sample_train_and_test.out_traindata
        # ------------------------------------------------------------------------
        create_sparse_test_dataset = self.new_task('create_sparse_test_dataset', CreateSparseTestDataset,
                dataset_name = self.dataset_name,
                replicate_id = self.replicate_id,
                slurminfo = sl.SlurmInfo(
                    runmode=sl.RUNMODE_HPC, # For debugging
                    project=self.slurm_project,
                    partition='devcore',
                    cores='2',
                    time='15:00',
                    jobname='MMSampleTrainTest',
                    threads='2'
                ))
        create_sparse_test_dataset.in_testdata = sample_train_and_test.out_testdata
        create_sparse_test_dataset.in_signatures = create_sparse_train_dataset.out_signatures
        # ------------------------------------------------------------------------
        ungzip_testdata = self.new_task('ungzip_testdata', UnGzipFile)
        ungzip_testdata.in_gzipped = create_sparse_test_dataset.out_sparse_testdata
        # ------------------------------------------------------------------------
        ungzip_traindata = self.new_task('ungzip_testdata', UnGzipFile)
        ungzip_traindata.in_gzipped = create_sparse_train_dataset.out_sparse_traindata
        # ------------------------------------------------------------------------
        train_lin_model = self.new_task('train_lin_model', TrainLinearModel,
                replicate_id = self.replicate_id,
                train_size = self.train_size,
                lin_type = self.lin_type,
                lin_cost = self.lin_cost,
                dataset_name = self.dataset_name,
                slurminfo = sl.SlurmInfo(
                    runmode=sl.RUNMODE_HPC, # For debugging
                    project=self.slurm_project,
                    partition='devcore',
                    cores='2',
                    time='15:00',
                    jobname='MMSampleTrainTest',
                    threads='2'
                ))
        train_lin_model.in_traindata = ungzip_traindata.out_ungzipped
        # ------------------------------------------------------------------------
        predict_lin = self.new_task('predict_lin', PredictLinearModel,
                dataset_name = self.dataset_name,
                replicate_id = self.replicate_id,
                slurminfo = sl.SlurmInfo(
                    runmode=sl.RUNMODE_HPC, # For debugging
                    project=self.slurm_project,
                    partition='devcore',
                    cores='2',
                    time='15:00',
                    jobname='MMSampleTrainTest',
                    threads='2'
                ))
        predict_lin.in_linmodel = train_lin_model.out_linmodel
        predict_lin.in_sparse_testdata = ungzip_testdata.out_ungzipped
        # ------------------------------------------------------------------------
        assess_linear = self.new_task('assess_linear', AssessLinearRMSD,
                dataset_name = self.dataset_name,
                replicate_id = self.replicate_id,
                lin_cost = self.lin_cost,
                slurminfo = sl.SlurmInfo(
                    runmode=sl.RUNMODE_HPC, # For debugging
                    project=self.slurm_project,
                    partition='devcore',
                    cores='2',
                    time='15:00',
                    jobname='MMSampleTrainTest',
                    threads='2'
                ))
        assess_linear.in_prediction = predict_lin.out_prediction
        assess_linear.in_linmodel = ungzip_traindata.out_ungzipped
        assess_linear.in_sparse_testdata = ungzip_testdata.out_ungzipped
        return assess_linear

# ====================================================================================================

if __name__ == '__main__':
    sl.run_local()
