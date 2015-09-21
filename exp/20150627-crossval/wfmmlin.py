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
    sampling_seed = luigi.Parameter(default=None)
    sampling_method = luigi.Parameter()
    test_size = luigi.Parameter()
    train_size = luigi.Parameter()
    lin_type = luigi.Parameter()
    slurm_project = luigi.Parameter()
    parallel_lin_train = luigi.BooleanParameter()
    runmode = luigi.Parameter()
    #folds_count = luigi.Parameter()

    def workflow(self):
        if self.runmode == 'local':
            runmode = sl.RUNMODE_LOCAL
        elif self.runmode == 'hpc':
            runmode = sl.RUNMODE_HPC
        elif self.runmode == 'mpi':
            runmode = sl.RUNMODE_MPI
        else:
            raise Exception('Runmode is none of local, hpc, nor mpi. Please fix and try again!')

        return_tasks = []
        '''
        The dependency graph is defined here!
        '''
        # --------------------------------------------------------------------------------
        existing_smiles = self.new_task('existing_smiles', ExistingSmiles,
                dataset_name = self.dataset_name)
        # --------------------------------------------------------------------------------
        gen_sign_filter_subst = self.new_task('gen_sign_filter_subst', GenerateSignaturesFilterSubstances,
                min_height = 1,
                max_height = 3,
                dataset_name = self.dataset_name,
                slurminfo = sl.SlurmInfo(
                    runmode=runmode,
                    project=self.slurm_project,
                    partition='core',
                    cores='8',
                    time='1:00:00',
                    jobname='MMLinGenSign',
                    threads='8'
                ))
        gen_sign_filter_subst.in_smiles = existing_smiles.out_smiles
        # --------------------------------------------------------------------------------
        for replicate_id in ['r1']:
            create_unique_sign_copy = self.new_task('create_unique_sign_copy_%s' % replicate_id, CreateReplicateCopy,
                    replicate_id = replicate_id)
            create_unique_sign_copy.in_file = gen_sign_filter_subst.out_signatures
            # --------------------------------------------------------------------------------
            for test_size in ['50000']:
                # --------------------------------------------------------------------------------
                for train_size in ['100']: # ['100', '1000', ...
                    sample_train_and_test = self.new_task('sample_trn%s_tst%s' % (train_size, test_size), SampleTrainAndTest,
                            seed = self.sampling_seed,
                            test_size = test_size,
                            train_size = train_size,
                            sampling_method = self.sampling_method,
                            dataset_name = self.dataset_name,
                            replicate_id = replicate_id,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='devcore',
                                cores='12',
                                time='1:00:00',
                                jobname='MMLinSampleTrainTest',
                                threads='1'
                            ))
                    sample_train_and_test.in_signatures = create_unique_sign_copy.out_copy
                    # --------------------------------------------------------------------------------
                    create_sparse_train_dataset = self.new_task('create_sparse_traindata_trn%s_tst%s' % (train_size, test_size), CreateSparseTrainDataset,
                            dataset_name = self.dataset_name,
                            replicate_id = replicate_id,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='node',
                                cores='16',
                                time='1-00:00:00',
                                jobname='MMLinCreateSparseTrain',
                                threads='16'
                            ))
                    create_sparse_train_dataset.in_traindata = sample_train_and_test.out_traindata
                    # ------------------------------------------------------------------------
                    create_sparse_test_dataset = self.new_task('create_sparse_testdata_trn%s_tst%s' % (train_size, test_size), CreateSparseTestDataset,
                            dataset_name = self.dataset_name,
                            replicate_id = replicate_id,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='node',
                                cores='16',
                                time='1-00:00:00',
                                jobname='MMLinCreateSparseTest',
                                threads='16'
                            ))
                    create_sparse_test_dataset.in_testdata = sample_train_and_test.out_testdata
                    create_sparse_test_dataset.in_signatures = create_sparse_train_dataset.out_signatures
                    # ------------------------------------------------------------------------
                    ungzip_testdata = self.new_task('ungzip_testdata_trn%s_tst%s' % (train_size, test_size), UnGzipFile,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='1:00:00',
                                jobname='MMLinUnGzipTestData',
                                threads='1'
                            ))
                    ungzip_testdata.in_gzipped = create_sparse_test_dataset.out_sparse_testdata
                    # ------------------------------------------------------------------------
                    ungzip_traindata = self.new_task('ungzip_traindata_trn%s_tst%s' % (train_size, test_size), UnGzipFile,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='1:00:00',
                                jobname='MMLinUnGzipTrainData',
                                threads='1'
                            ))
                    ungzip_traindata.in_gzipped = create_sparse_train_dataset.out_sparse_traindata
                    # ------------------------------------------------------------------------
                    train_lin_model = self.new_task('train_lin_trn%s_tst%s' % (train_size, test_size), TrainLinearModel,
                            replicate_id = replicate_id,
                            train_size = train_size,
                            lin_type = self.lin_type,
                            lin_cost = self.lin_cost,
                            dataset_name = self.dataset_name,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='4-00:00:00',
                                jobname='MMTrainLinear',
                                threads='1'
                            ))
                    train_lin_model.in_traindata = ungzip_traindata.out_ungzipped
                    # ------------------------------------------------------------------------
                    predict_lin = self.new_task('predict_lin_trn%s_tst%s' % (train_size, test_size), PredictLinearModel,
                            dataset_name = self.dataset_name,
                            replicate_id = replicate_id,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='4:00:00',
                                jobname='MMSampleTrainTest',
                                threads='1'
                            ))
                    predict_lin.in_linmodel = train_lin_model.out_linmodel
                    predict_lin.in_sparse_testdata = ungzip_testdata.out_ungzipped
                    # ------------------------------------------------------------------------
                    assess_linear = self.new_task('assess_lin_trn%s_tst%s' % (train_size, test_size), AssessLinearRMSD,
                            dataset_name = self.dataset_name,
                            replicate_id = replicate_id,
                            lin_cost = self.lin_cost,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='15:00',
                                jobname='MMSampleTrainTest',
                                threads='1'
                            ))
                    assess_linear.in_prediction = predict_lin.out_prediction
                    assess_linear.in_linmodel = ungzip_traindata.out_ungzipped
                    assess_linear.in_sparse_testdata = ungzip_testdata.out_ungzipped
                    return_tasks.append(assess_linear)
        return return_tasks

# ====================================================================================================

if __name__ == '__main__':
    sl.run_local()
