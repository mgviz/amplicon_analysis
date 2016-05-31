#!/usr/bin/env python

import argparse
import os
import time
import logging as log
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from string import Template
from subprocess import Popen, PIPE

_DEF_COV_THRESHOLD = 100
_DATA_FOLDER = 'data'
_SAMPLE_DATA = 'SampleData.csv'
_SAMPLE_SELECTION = 'SampleSelection.csv'

# Output folders
_ALIGNMENT_FOLDER = 'Alignment'
_COV_FOLDER = 'covs'
_PLOT_FOLDER = 'plots'
_REPORT_FOLDER = 'reports'
_BED_FOLDER = 'beds'

_MERGED_COV_FILE = 'all_samples.perbase.cov'

# Required BEDtools v.2.19.0 or above!
_BEDTOOLS_COVPERBASE_CMD = ('coverageBed -d -a $bed -b $bam' +
                            ' | grep -v \'^all\' > $out')


def _setup_argparse():
    """It prepares the command line argument parsing"""

    desc = 'description'
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=formatter_class)
    parser.add_argument('-p', '--project', dest='project_fpath', required=True,
                        help='Project folder')
    parser.add_argument('-v', '--verbose', dest='verbosity',
                        help='increase output verbosity', action='store_true')
    parser.add_argument('-t', '--cov_threshold', dest='cov_threshold',
                        help='Coverage threshold', default=_DEF_COV_THRESHOLD,
                        type=int)

    args = parser.parse_args()
    return args


def _get_options():
    """It checks arguments values"""
    args = _setup_argparse()

    # Setting up logging system
    if args.verbosity:
        log.basicConfig(format="[%(levelname)s] %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="[%(levelname)s] %(message)s", level=log.ERROR)

    # Checking if input file is provided
    project_absfpath = os.path.abspath(args.project_fpath)
    if not os.path.isdir(project_absfpath):
        raise IOError('Project folder does not exist. Check path.')
    else:
        args.project_fpath = project_absfpath

        # Checking if input file is provided
        data_fpath = os.path.join(project_absfpath, _DATA_FOLDER)
        args.sample_data_fpath = os.path.join(data_fpath, _SAMPLE_DATA)
        args.sample_selec_fpath = os.path.join(data_fpath, _SAMPLE_SELECTION)
        if not os.path.isfile(args.sample_data_fpath):
            raise IOError('SampleData file does not exist in "' +
                          data_fpath + '".')
        if not os.path.isfile(args.sample_selec_fpath):
            raise IOError('SampleSelection file does not exist in "' +
                          data_fpath + '".')

    # Checking if coverage threshold is a positive integer
    if (not isinstance(args.cov_threshold, int)) or args.cov_threshold < 0:
        raise IOError('Coverage threshold must be a positive integer.')

    return args


def _get_time(fancy=True):
    """Timestamp"""
    if fancy:
        return time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return time.strftime("%Y-%m-%d_%H-%M-%S")


def parse_cov_file(fpath, sep='\t'):
    """It reads a TSV file into a pandas dataframe
    :param fpath: TSV file path
    :param sep: field delimiter character
    """
    header = ['ref', 'start', 'end', 'feature', 'base', 'coverage', 'sample']
    df = pd.read_csv(fpath, sep=sep, header=None)
    df['sample'] = os.path.splitext(os.path.basename(fpath))[0]

    # Checking that header and dataframe columns coincide in number
    try:
        assert (len(header) == len(df.columns))
    except AssertionError:
        log.error('File "' + fpath + '" has an incorrect number of columns.')

    df.columns = header

    return df


def cov_plot(df, out_folder, cov_threshold=None, feats=None, samps=None):
    """It plots the coverage per base for each sample for a specific reference
    :param df: input dataframe
    :param out_folder: output folder to store the plots
    :param cov_threshold: coverage threshold
    :param feats: list of features to plot
    :param samps: list of samples to plot
    """

    if not feats:
        features = list(set(df['feature']))
    else:
        features = feats

    for feature in features:

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        df_feature = df[df['feature'] == feature]

        if not samps:
            samples = list(set(df_feature['sample']))
        else:
            samples = samps

        sns.set_style("darkgrid")

        # Plotting a line for each sample
        for sample in samples:
            sample_data = df_feature[df_feature['sample'] == sample]
            ax1.plot(sample_data['base'], sample_data['coverage'],
                     color='black', alpha=0.5)

            # Setting plot limits
            ax1.set_ylim(0, df_feature['coverage'].max() + 50)
            ax1.set_xlim(0, df_feature['base'].max() + 1)

            # Plotting coverage threshold
            if cov_threshold:
                ax1.hlines(y=cov_threshold, xmin=ax1.get_xlim()[0],
                           xmax=ax1.get_xlim()[1], color='r')

                # Checking if there are intersections with the cov threshold
                below = sample_data[sample_data['coverage'] < cov_threshold]
                above = sample_data[sample_data['coverage'] > cov_threshold]
                if (not below.empty) and (not above.empty):
                    # Getting intersections
                    intersections = []
                    x = sample_data['coverage'].tolist()
                    for i, couple in enumerate(zip(x[0:], x[1:])):
                        if couple[0] < cov_threshold <= couple[1]:
                            point = sample_data.iloc[[i + 1]]['base'].values[0]
                            intersections.append(point)
                        if couple[0] >= cov_threshold > couple[1]:
                            point = sample_data.iloc[[i]]['base'].values[0]
                            intersections.append(point)

                    if intersections:
                        ax2 = ax1.twiny()
                        ax2.set_xlim(ax1.get_xlim())
                        ax2.set_ylim(ax1.get_ylim())
                        ax2.vlines(x=intersections, ymin=cov_threshold,
                                   ymax=ax2.get_ylim()[1], color='black',
                                   linestyle='--')
                        ax2.grid(b=False)
                        ax2.set_xticks(intersections)

            # Customizing labels
            fig.suptitle(feature)
            ax1.set_xlabel('Position (bp)')
            ax1.set_ylabel('Coverage')

            # Saving plot and clearing it
            figname = sample + '-' + feature + '.pbcov.png'
            fig.savefig(os.path.join(out_folder, figname))
            #plt.clf()
            plt.close(fig)


def percentage(part, whole):
    """It computes percentages
    :param part: part of the data
    :param whole: total data
    """
    # Avoiding ZeroDivision error and returning negative number if so
    if whole > 0:
        return round(100 * float(part) / float(whole), 2)
    else:
        return -1


def _get_cov_stats(df, cov_threshold=None):
    """Gets stats from a coverage dataframe"""
    # Summarizing data
    # http://bconnelly.net/2013/10/summarizing-data-in-python-with-pandas/
    df_cov = df.get(['sample', 'feature', 'coverage'])
    df_grouped = df_cov.groupby(['sample', 'feature'])
    stats = df_grouped.agg([np.size, np.min, np.max]).reset_index()
    stats.columns = ['sample', 'feature', 'length', 'cov_min', 'cov_max']

    # If there is coverage threshold, add coverage breadth
    if cov_threshold:
        col_name = '%cov_breadth_' + str(cov_threshold) + 'x'
        stats[col_name] = df_grouped.agg([lambda x: percentage(
                np.size(np.where(x > cov_threshold)),
                np.size(x))]).reset_index().iloc[:, -1].values

    return stats


def _write_stats_to_excel(stats, out_fpath, cov_threshold=None):
    """Write coverage stats to an excel file"""
    # Ordering columns
    if cov_threshold:
        stats = stats[['sample', 'feature', 'length', 'cov_min', 'cov_max',
                       '%cov_breadth_' + str(cov_threshold) + 'x']]
    else:
        stats = stats[['sample', 'feature', 'length', 'cov_min', 'cov_max']]

    # Writing to excel
    # http://xlsxwriter.readthedocs.org/working_with_pandas.html
    writer = pd.ExcelWriter(os.path.join(out_fpath, 'stats.xlsx'),
                            engine='xlsxwriter')
    stats.to_excel(writer, sheet_name='stats', index=False)

    # If there is coverage threshold, add conditional formatting
    if cov_threshold:
        workbook = writer.book
        worksheet = writer.sheets['stats']

        # Defining formats
        green_format = workbook.add_format({'bg_color': '#C6EFCE'})
        red_format = workbook.add_format({'bg_color': '#FFC7CE'})
        orange_format = workbook.add_format({'bg_color': '#FFD27F'})

        # Applying formats to cell range
        cell_range = 'F2:F' + str(len(stats.index) + 1)
        worksheet.conditional_format(cell_range, {'type': 'cell',
                                                  'criteria': 'equal to',
                                                  'value': 100,
                                                  'format': green_format})
        worksheet.conditional_format(cell_range, {'type': 'cell',
                                                  'criteria': 'equal to',
                                                  'value': 0,
                                                  'format': red_format})
        worksheet.conditional_format(cell_range, {'type': 'cell',
                                                  'criteria': 'between',
                                                  'minimum': 0,
                                                  'maximum': 100,
                                                  'format': orange_format})

    writer.save()


def _create_folder(folder):
    """Creates a new folder given a path
    :param folder: path of the folder
    """
    if os.path.exists(folder):
        log.warning('Folder "' + folder + '" already exists.')
    else:
        try:
            os.makedirs(folder)
            log.debug('Creating folder "' + folder + '".')
        except:
            raise IOError('Unable to create output folders. Check permissions.')


def run_bedtools_get_cov(samples, bam_out_fpath, bed_out_fpath, cov_out_fpath,
                         cmd):
    """Runs a bedtools getCoverage command
    :param samples: names of the samples
    :param bam_out_fpath: path of the input BAM file
    :param bed_out_fpath: path of the input BED file
    :param cov_out_fpath: path of the output coverage file
    :param cmd: template of the command
    """
    template = Template(cmd)
    for sample in samples:
        bam = os.path.join(bam_out_fpath, sample + '.bam')
        bed = os.path.join(bed_out_fpath, sample + '.bed')
        out = os.path.join(cov_out_fpath, sample + '.pbcov')
        cmd = template.substitute(bam=bam, bed=bed, out=out)
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        output = p.communicate()[1]
        if p.returncode != 0:
            raise RuntimeError('Failed BEDtools command "' + cmd + '". ' +
                               output)


def concatenate_files(files, out_fpath):
    """It concatenates multiple files into one file
    :param files: paths of the input files
    :param out_fpath: path of the output file
    """
    out_fhand = open(out_fpath, 'w')

    for f in files:
        in_fhand = open(f, 'r')
        fname = os.path.splitext(os.path.basename(f))[0]
        for line in in_fhand:
            line = line.strip() + '\t' + fname + '\n'
            out_fhand.write(line)
        in_fhand.close()

    out_fhand.flush()
    out_fhand.close()


def main():
    """The main function"""

    # Parsing options
    options = _get_options()
    if options.verbosity:
        log.info('START "' + _get_time() + '".')
        log.debug('Options parsed: "' + str(options) + '".')

    # Setting up output folders paths
    out_folders = {
        'bam_folder': os.path.join(options.project_fpath, _ALIGNMENT_FOLDER),
        'bed_folder': os.path.join(options.project_fpath, _BED_FOLDER),
        'cov_folder': os.path.join(options.project_fpath, _COV_FOLDER),
        'plot_folder': os.path.join(options.project_fpath, _PLOT_FOLDER),
        'report_folder': os.path.join(options.project_fpath, _REPORT_FOLDER)}

    # Creating output folders
    log.info('Creating output folders...')
    for value in out_folders.values():
        _create_folder(value)

    # Retrieving desired sample names
    sample_selec_fhand = open(options.sample_selec_fpath, 'r')
    samples = [sample.strip() for sample in sample_selec_fhand]
    samples = filter(None, samples)  # Removing empty lines
    sample_selec_fhand.close()
    log.debug('Samples specified: "' + str(samples) + '".')

    # Checking if there is a BAM file for each specified sample
    # Also creating a ordered BAM file list depending on samples list order
    bam_files = [f for f in os.listdir(out_folders['bam_folder']) if
                 f.endswith('.bam')]
    log.debug('BAM files found: "' + str(bam_files) + '".')
    samples_with_bam = []
    bam_samples = []
    for sample in samples:
        for bam in bam_files:
            if bam.startswith(sample + '_S'):
                samples_with_bam.append(sample)
                bam_samples.append(bam)
                break
    samples_without_bam = list(set(samples) - set(samples_with_bam))
    if len(samples_without_bam) != 0:
        raise ValueError('No BAM files for samples: "' +
                         str(samples_without_bam) + '".')

    # Creating a BED file for each desired sample
    log.info('Creating BED files...')
    sample_data_df = pd.read_csv(options.sample_data_fpath, sep='\t', header=0)
    desired_columns = ['chromosome', 'amplicon_start', 'amplicon_end',
                       'amplicon_name']
    for i, sample in enumerate(samples):
        subselect = sample_data_df[desired_columns][(
            sample_data_df.sample_name == sample)]

        # Checking if there are regions associated to sample
        if subselect.empty:
            msg = ('No region found for sample "' + sample +
                   '" in SampleData.csv.')
            raise ValueError(msg)

        # Checking if region end is higher than region start
        wrong_regions = subselect.loc[subselect['amplicon_end'] -
                                      subselect['amplicon_start'] <= 0]
        if not wrong_regions.empty:
            msg = 'Region end has to be higher than region start:\n' +\
                  str(wrong_regions)
            raise ValueError(msg)

        bed_fname = os.path.splitext(bam_samples[i])[0] + '.bed'
        bed_fpath = os.path.join(out_folders['bed_folder'], bed_fname)
        if os.path.isfile(bed_fpath):
            log.warning('File "' + bed_fpath + '" already exists. Overwriting.')
        subselect.to_csv(bed_fpath, sep='\t', index=False, header=False)

    # Running BEDtools
    log.info('Running BEDtools...')
    inds = map(lambda x: os.path.splitext(x)[0], bam_samples)
    run_bedtools_get_cov(inds, out_folders['bam_folder'],
                         out_folders['bed_folder'], out_folders['cov_folder'],
                         _BEDTOOLS_COVPERBASE_CMD)

    # Creating coverage plots and getting stats
    log.info('Generating coverage plots...')
    cov_fnames = [f for f in os.listdir(out_folders['cov_folder']) if
                 f.endswith('.pbcov')]
    log.debug('Coverage files found: "' + str(cov_fnames) + '".')
    stats_all = pd.DataFrame()
    for cov_fname in cov_fnames:
        # Parsing input file
        cov_fpath = os.path.join(out_folders['cov_folder'], cov_fname)
        df = parse_cov_file(cov_fpath)
        # Plotting
        cov_plot(df, out_folders['plot_folder'], options.cov_threshold)

        stats = _get_cov_stats(df, options.cov_threshold)
        stats_all = stats_all.append(stats)

    # Creating excel with statistics
    _write_stats_to_excel(stats_all, out_folders['report_folder'],
                          options.cov_threshold)

    if options.verbosity:
        log.info('END "' + _get_time() + '".')


if __name__ == '__main__':
    main()
