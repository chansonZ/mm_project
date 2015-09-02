from mmcomp import *
import luigi
import sciluigi as sl
import time

# ====================================================================================================
#  New components for Cross-validation - May 8, 2015
# ====================================================================================================

class CrossValidate(sl.WorkflowTask):
    task = luigi.Parameter()
    '''
    For now, a sketch on how to implement Cross-Validation as a sub-workflow components
    '''

    # PARAMETERS
    dataset_name = luigi.Parameter(default='mm_test_small')
    folds_count = luigi.IntParameter()
    replicate_id = luigi.Parameter()
    min_height = luigi.Parameter()
    max_height = luigi.Parameter()
    test_size = luigi.Parameter(default='10')
    train_size = luigi.Parameter(default='50')
    slurm_project = luigi.Parameter(default='b2013262')

    def workflow(self):
        # Initialize tasks
        mmtestdata = self.new_task('mmtestdata', ExistingSmiles,
                replicate_id=self.replicate_id,
                dataset_name=self.dataset_name)
        gensign = self.new_task('gensign', GenerateSignaturesFilterSubstances,
                replicate_id=self.replicate_id,
                min_height = self.min_height,
                max_height = self.max_height,
                slurminfo = sl.SlurmInfo(
                    runmode=sl.RUNMODE_HPC, # For debugging
                    project=self.slurm_project,
                    partition='devcore',
                    cores='2',
                    time='15:00',
                    jobname='MMGenSignTest',
                    threads='2'
                ))
        replcopy = self.new_task('replcopy', CreateReplicateCopy,
                replicate_id=self.replicate_id)
        samplett = self.new_task('sampletraintest', SampleTrainAndTest,
                replicate_id=self.replicate_id,
                sampling_method='random',
                seed='1',
                test_size=self.test_size,
                train_size=self.train_size,
                slurminfo = sl.SlurmInfo(
                    runmode=sl.RUNMODE_HPC, # For debugging
                    project='b2013262',
                    partition='devcore',
                    cores='2',
                    time='15:00',
                    jobname='MMSampleTrainTest',
                    threads='2'
                ))
        sprstrain = self.new_task('sparsetrain', CreateSparseTrainDataset,
                replicate_id=self.replicate_id,
                slurminfo = sl.SlurmInfo(
                    runmode=sl.RUNMODE_HPC, # For debugging
                    project=self.slurm_project,
                    partition='devcore',
                    cores='2',
                    time='15:00',
                    jobname='MMSampleTrainTest',
                    threads='2'
                ))
        gunzip = self.new_task('gunzip_sparsetrain', UnGzipFile)

        # Connect tasks by their inports and outports
        gensign.in_smiles = mmtestdata.out_smiles
        replcopy.in_file = gensign.out_signatures
        samplett.in_signatures = replcopy.out_copy
        sprstrain.in_traindata = samplett.out_traindata
        gunzip.in_gzipped = sprstrain.out_sparsetraindata

        tasks = {}
        costseq = [str(int(10**p)) for p in xrange(1,9)]
        for cost in costseq:
            tasks[cost] = {}
            # Branch the workflow into one branch per fold
            for fold_idx in xrange(self.folds_count):
                # Init tasks
                create_folds = self.new_task('create_fold_%d' % fold_idx, CreateFolds,
                        fold_index = fold_idx,
                        folds_count = self.folds_count,
                        seed = 0.637)
                train_lin = self.new_task('trainlin_fold_%d_cost_%s' % (fold_idx, cost), TrainLinearModel,
                        replicate_id = self.replicate_id,
                        lin_type = '0', # 0 = Regression
                        lin_cost = cost,
                        slurminfo = sl.SlurmInfo(
                            runmode=sl.RUNMODE_LOCAL, # For debugging
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='15:00',
                            jobname='trnlin_f%02d_c%010d' % (fold_idx, int(cost)),
                            threads='1'
                        ))
                pred_lin = self.new_task('predlin_fold_%d_cost_%s' % (fold_idx, cost), PredictLinearModel,
                        replicate_id = self.replicate_id,
                        slurminfo = sl.SlurmInfo(
                            runmode=sl.RUNMODE_LOCAL, # For debugging
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='15:00',
                            jobname='predlin_f%02d_c%010d' % (fold_idx, int(cost)),
                            threads='1'
                        ))
                assess_lin = self.new_task('assesslin_fold_%d_cost_%s' % (fold_idx, cost), AssessLinearRMSD,
                        lin_cost = cost,
                        slurminfo = sl.SlurmInfo(
                            runmode=sl.RUNMODE_LOCAL, # For debugging
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='15:00',
                            jobname='assesslin_f%02d_c%010d' % (fold_idx, int(cost)),
                            threads='1'
                        ))

                # Connect tasks
                create_folds.in_dataset = gunzip.out_ungzipped
                train_lin.in_traindata = create_folds.out_traindata
                pred_lin.in_linmodel = train_lin.out_linmodel
                pred_lin.in_sparse_testdata = create_folds.out_testdata
                assess_lin.in_linmodel = train_lin.out_linmodel
                assess_lin.in_sparse_testdata = create_folds.out_testdata
                assess_lin.in_prediction = pred_lin.out_prediction

                tasks[cost][fold_idx] = {}
                tasks[cost][fold_idx]['create_folds'] = create_folds
                tasks[cost][fold_idx]['train_linear'] = train_lin
                tasks[cost][fold_idx]['predict_linear'] = pred_lin
                tasks[cost][fold_idx]['assess_linear'] = assess_lin

            # Calculate the average RMSD for each cost value
            average_rmsd = self.new_task('average_rmsd_cost_%s' % cost, CalcAverageRMSDForCost,
                    lin_cost=cost)
            average_rmsd.in_assessments = [tasks[cost][fold_idx]['assess_linear'].out_assessment for fold_idx in xrange(self.folds_count)]

            tasks[cost]['average_rmsd'] = average_rmsd

        average_rmsds = [tasks[cost]['average_rmsd'] for cost in costseq]

        sel_lowest_rmsd = self.new_task('select_lowest_rmsd', SelectLowestRMSD)
        sel_lowest_rmsd.in_values = [average_rmsd.out_rmsdavg for average_rmsd in average_rmsds]

        return sel_lowest_rmsd

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
        train_lin_model.in_traindata = create_sparse_train_dataset.out_sparse_traindata
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
        predict_lin.in_sparse_testdata = create_sparse_test_dataset.out_sparse_testdata
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
        assess_linear.in_linmodel = train_lin_model.out_linmodel
        assess_linear.in_sparse_testdata = create_sparse_test_dataset.out_sparse_testdata
        return assess_linear

# ====================================================================================================

if __name__ == '__main__':
    sl.run_local()
