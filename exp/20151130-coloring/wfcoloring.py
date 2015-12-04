import sciluigi as sl
import luigi
import mmcomp as mm

class ColoringWorkflow(sl.WorkflowTask):
    runmode = luigi.Parameter()
    slurm_project = luigi.Parameter()

    def workflow(self):
        # ------------------------------------------------------------------------
        dataset_name = 'acd_logd'
        replicate_id = 'r1'
        # ------------------------------------------------------------------------
        existing_svm_model = self.new_task('existing_svmmdl',
            mm.ExistingFile,
            file_path='data/acd_logd.smi.h1_3.sign.r1.50000_80000_rand_trn.csr.ungz.g0p001_c100_s3_t2.svm')
        # ------------------------------------------------------------------------
        existing_traindata_ungzipped = self.new_task('existing_traindata',
            mm.ExistingFile,
            file_path='data/acd_logd.smi.h1_3.sign.r1.50000_80000_rand_trn.csr.ungz')
        # ------------------------------------------------------------------------
        predict_train = self.new_task('predict_train',
                mm.PredictSVMModel,
                dataset_name = dataset_name,
                replicate_id = replicate_id,
                slurminfo = sl.SlurmInfo(
                    runmode=self.runmode,
                    project=self.slurm_project,
                    partition='core',
                    cores='1',
                    time='4:00:00',
                    jobname='predict_train',
                    threads='1'
                ))
        predict_train.in_svmmodel = existing_svm_model.out_file
        predict_train.in_sparse_testdata = existing_traindata_ungzipped.out_file
        # ------------------------------------------------------------------------
        select_idx10 = self.new_task('select_idx10',
                mm.SelectPercentIndexValue,
                percent_index=10)
        select_idx10.in_prediction = predict_train.out_prediction
        # ------------------------------------------------------------------------
        select_idx90 = self.new_task('select_idx90',
                mm.SelectPercentIndexValue,
                percent_index=90)
        select_idx90.in_prediction = predict_train.out_prediction
        # ------------------------------------------------------------------------
        return [select_idx10, select_idx90]

if __name__ == '__main__':
    sl.run_local(main_task_cls=ColoringWorkflow)
