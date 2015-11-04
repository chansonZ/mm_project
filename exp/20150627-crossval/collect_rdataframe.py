import glob
import sciluigi as sl
import csv
import re

dataframe = []
def main():
    glob_pattern = 'data/*ungz.s12*.rmsd'
    rmsd_paths = glob.glob(glob_pattern)
    for rp in rmsd_paths:
        datarow = {}
        ms = re.match('data/(solubility|acd_logd).smi.h1_3.sign.(r[0-9]).([0-9]+)_([0-9]+|rest)_rand_trn.csr.ungz.s12_c([0-9\.]+).(lin|svm)mdl.pred.rmsd', rp)
        m = ms.groups()
        datarow['dataset'] = m[0]
        datarow['replicate'] = m[1]
        datarow['test_size'] = m[2]
        datarow['training_size'] = m[3]
        datarow['cost'] = m[4]
        if m[5] == 'lin':
            datarow['learning_method'] = 'liblinear'
        elif m[5] == 'svm':
            datarow['learning_method'] = 'svm'
        with open(rp) as rf:
            rd = sl.util.recordfile_to_dict(rf)
            datarow['rmsd'] = rd['rmsd']
            datarow['cost'] = rd['cost']
            dataframe.append(datarow)
    with open('rdataframe.csv', 'w') as fh:
        csvwrt = csv.writer(fh)
        csvwrt.writerow(['dataset', 'learning_method', 'training_size', 'replicate', 'rmsd', 'model_creation_time', 'cost'])
        for row in dataframe:
            csvwrt.writerow([row['dataset'], row['learning_method'], row['training_size'], row['replicate'], row['rmsd'], 'N/A', row['cost']])

if __name__ == '__main__':
    main()
