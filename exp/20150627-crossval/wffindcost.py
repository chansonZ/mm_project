from mmcomp import *
import logging
import luigi
import sciluigi as sl
import time

# ====================================================================================================
#  New components for Cross-validation - May 8, 2015
# ====================================================================================================

log = logging.getLogger('sciluigi-interface')

class CrossValidate(sl.WorkflowTask):
    '''
    For now, a sketch on how to implement Cross-Validation as a sub-workflow components
    '''

    # PARAMETERS
    dataset_name = luigi.Parameter()
    folds_count = luigi.IntParameter()
    replicate_id = luigi.Parameter()
    min_height = luigi.Parameter()
    max_height = luigi.Parameter()
    test_size = luigi.Parameter()
    train_size = luigi.Parameter()
    randomdatasize_mb = luigi.IntParameter()
    slurm_project = luigi.Parameter(default='b2013262')
    runmode = luigi.Parameter()

    def workflow(self):
        if self.runmode == 'local':
            runmode = sl.RUNMODE_LOCAL
        elif self.runmode == 'hpc':
            runmode = sl.RUNMODE_HPC
        elif self.runmode == 'mpi':
            runmode = sl.RUNMODE_MPI
        else:
            raise Exception('Runmode is none of local, hpc, nor mpi. Please fix and try again!')

        # ----------------------------------------------------------------
        mmtestdata = self.new_task('mmtestdata', ExistingSmiles,
                replicate_id=self.replicate_id,
                dataset_name=self.dataset_name)
        # ----------------------------------------------------------------
        gensign = self.new_task('gensign', GenerateSignaturesFilterSubstances,
                replicate_id=self.replicate_id,
                min_height = self.min_height,
                max_height = self.max_height,
                slurminfo = sl.SlurmInfo(
                    runmode=runmode,
                    project=self.slurm_project,
                    partition='core',
                    cores='8',
                    time='1:00:00',
                    jobname='mmgensign',
                    threads='8'
                ))
        gensign.in_smiles = mmtestdata.out_smiles
        # ----------------------------------------------------------------
        replcopy = self.new_task('replcopy', CreateReplicateCopy,
                replicate_id=self.replicate_id)
        replcopy.in_file = gensign.out_signatures
        lowest_rmsds = []
        for train_size in [str(i) for i in [100, 1000, 5000, 10000, 20000, 80000, 160000, 320000, 'rest']]:
            samplett = self.new_task('sampletraintest_%s' % train_size, SampleTrainAndTest,
                    replicate_id=self.replicate_id,
                    sampling_method='random',
                    seed='1',
                    test_size=self.test_size,
                    train_size=train_size,
                    slurminfo = sl.SlurmInfo(
                        runmode=runmode,
                        project='b2013262',
                        partition='core',
                        cores='12',
                        time='1:00:00',
                        jobname='mmsampletraintest_%s' % train_size,
                        threads='1'
                    ))
            samplett.in_signatures = replcopy.out_copy
            # ----------------------------------------------------------------
            sprstrain = self.new_task('sparsetrain_%s' % train_size, CreateSparseTrainDataset,
                    replicate_id=self.replicate_id,
                    slurminfo = sl.SlurmInfo(
                        runmode=runmode,
                        project=self.slurm_project,
                        partition='node',
                        cores='16',
                        time='1-00:00:00', # Took ~16hrs for acd_logd, size: rest(train) - 50000(test)
                        jobname='mmsparsetrain_%s' % train_size,
                        threads='16'
                    ))
            sprstrain.in_traindata = samplett.out_traindata
            # ----------------------------------------------------------------
            gunzip = self.new_task('gunzip_sparsetrain_%s' % train_size, UnGzipFile,
                    slurminfo = sl.SlurmInfo(
                        runmode=runmode,
                        project=self.slurm_project,
                        partition='core',
                        cores='1',
                        time='1:00:00',
                        jobname='gunzip_sparsetrain_%s' % train_size,
                        threads='1'
                    ))
            gunzip.in_gzipped = sprstrain.out_sparse_traindata
            # ----------------------------------------------------------------
            cntlines = self.new_task('countlines_%s' % train_size, CountLines,
                    slurminfo = sl.SlurmInfo(
                        runmode=runmode,
                        project=self.slurm_project,
                        partition='core',
                        cores='1',
                        time='15:00',
                        jobname='gunzip_sparsetrain_%s' % train_size,
                        threads='1'
                    ))
            cntlines.in_file = gunzip.out_ungzipped
            # ----------------------------------------------------------------
            genrandomdata= self.new_task('genrandomdata_%s' % train_size, CreateRandomData,
                    size_mb=self.randomdatasize_mb,
                    replicate_id=self.replicate_id,
                    slurminfo = sl.SlurmInfo(
                        runmode=runmode,
                        project=self.slurm_project,
                        partition='core',
                        cores='1',
                        time='1:00:00',
                        jobname='genrandomdata_%s' % train_size,
                        threads='1'
                    ))
            genrandomdata.in_basepath = gunzip.out_ungzipped
            # ----------------------------------------------------------------
            shufflelines = self.new_task('shufflelines_%s' % train_size, ShuffleLines,
                    slurminfo = sl.SlurmInfo(
                        runmode=runmode,
                        project=self.slurm_project,
                        partition='core',
                        cores='1',
                        time='15:00',
                        jobname='shufflelines_%s' % train_size,
                        threads='1'
                    ))
            shufflelines.in_randomdata = genrandomdata.out_random
            shufflelines.in_file = gunzip.out_ungzipped
            # ----------------------------------------------------------------

            tasks = {}
            costseq = [str(int(10**p)) for p in xrange(1,12)]
            # Branch the workflow into one branch per fold
            for fold_idx in xrange(self.folds_count):
                tasks[fold_idx] = {}
                # Init tasks
                create_folds = self.new_task('create_fold%02d_%s' % (fold_idx, train_size), CreateFolds,
                        fold_index = fold_idx,
                        folds_count = self.folds_count,
                        seed = 0.637,
                        slurminfo = sl.SlurmInfo(
                            runmode=runmode,
                            project=self.slurm_project,
                            partition='core',
                            cores='1',
                            time='1:00:00',
                            jobname='create_fold%02d_%s' % (fold_idx, train_size),
                            threads='1'
                        ))
                for cost in costseq:
                    tasks[fold_idx][cost] = {}
                    create_folds.in_dataset = shufflelines.out_shuffled
                    create_folds.in_linecount = cntlines.out_linecount
                    # -------------------------------------------------
                    train_lin = self.new_task('trainlin_fold_%d_cost_%s_%s' % (fold_idx, cost, train_size), TrainLinearModel,
                            replicate_id = self.replicate_id,
                            lin_type = '12', # 12, See: https://www.csie.ntu.edu.tw/~cjlin/liblinear/FAQ.html
                            lin_cost = cost,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='4-00:00:00',
                                jobname='trnlin_f%02d_c%010d_%s' % (fold_idx, int(cost), train_size),
                                threads='1'
                            ))
                    train_lin.in_traindata = create_folds.out_traindata
                    # -------------------------------------------------
                    pred_lin = self.new_task('predlin_fold_%d_cost_%s_%s' % (fold_idx, cost, train_size), PredictLinearModel,
                            replicate_id = self.replicate_id,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='8:00:00',
                                jobname='predlin_f%02d_c%010d_%s' % (fold_idx, int(cost), train_size),
                                threads='1'
                            ))
                    pred_lin.in_linmodel = train_lin.out_linmodel
                    pred_lin.in_sparse_testdata = create_folds.out_testdata
                    # -------------------------------------------------
                    assess_lin = self.new_task('assesslin_fold_%d_cost_%s_%s' % (fold_idx, cost, train_size), AssessLinearRMSD,
                            lin_cost = cost,
                            slurminfo = sl.SlurmInfo(
                                runmode=runmode,
                                project=self.slurm_project,
                                partition='core',
                                cores='1',
                                time='15:00',
                                jobname='assesslin_f%02d_c%010d_%s' % (fold_idx, int(cost), train_size),
                                threads='1'
                            ))
                    assess_lin.in_linmodel = train_lin.out_linmodel
                    assess_lin.in_sparse_testdata = create_folds.out_testdata
                    assess_lin.in_prediction = pred_lin.out_prediction
                    # -------------------------------------------------
                    tasks[fold_idx][cost] = {}
                    tasks[fold_idx][cost]['create_folds'] = create_folds
                    tasks[fold_idx][cost]['train_linear'] = train_lin
                    tasks[fold_idx][cost]['predict_linear'] = pred_lin
                    tasks[fold_idx][cost]['assess_linear'] = assess_lin

            # Tasks for calculating average RMSD and finding the cost with lowest RMSD
            avgrmsd_tasks = {}
            for cost in costseq:
                # Calculate the average RMSD for each cost value
                average_rmsd = self.new_task('average_rmsd_cost_%s_%s' % (cost, train_size), CalcAverageRMSDForCost,
                        lin_cost=cost)
                average_rmsd.in_assessments = [tasks[fold_idx][cost]['assess_linear'].out_assessment for fold_idx in xrange(self.folds_count)]
                avgrmsd_tasks[cost] = average_rmsd

            sel_lowest_rmsd = self.new_task('select_lowest_rmsd_%s' % train_size, SelectLowestRMSD)
            sel_lowest_rmsd.in_values = [average_rmsd.out_rmsdavg for average_rmsd in avgrmsd_tasks.values()]

            # Collect one lowest rmsd per train size
            lowest_rmsds.append(sel_lowest_rmsd)

        return lowest_rmsds


# ================================================================================

if __name__ == '__main__':
    sl.run_local()
