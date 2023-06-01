use linya::{Bar, Progress};
use polars::prelude::*;
use rand::{prelude::*, rngs::StdRng};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    env,
    error::Error,
    ffi::OsString,
    fs::{create_dir_all, read_dir},
    path, process,
    sync::{Arc, Mutex},
};

static NSIM: usize = 1000;
static SIMULATE: bool = false;

fn get_nth_arg(n: usize) -> Result<OsString, Box<dyn Error>> {
    match env::args_os().nth(n) {
        None => Err(From::from("expected an argument, but got none")),
        Some(file_path) => Ok(file_path),
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let input_path: path::PathBuf = get_nth_arg(1)
        .unwrap_or("imputation_results/".into())
        .into();
    let exp_output_path: path::PathBuf =
        get_nth_arg(1).unwrap_or("computed_matches/".into()).into();

    create_dir_all(exp_output_path.clone())?;

    let rep_folders: Vec<path::PathBuf> = read_dir(input_path)
        .unwrap()
        .map(|x| x.unwrap().path())
        .collect();

    // set up a fixed global rng for replication purposes
    let mut global_rng = StdRng::seed_from_u64(123);
    let global_seeds = rep_folders
        .iter()
        .map(|_| global_rng.gen::<u64>())
        .collect::<Vec<_>>();

    let progress = Mutex::new(Progress::new());

    let _ = rep_folders
        .par_iter()
        .zip(global_seeds)
        .map(|(x, seed)| {
            let rep = x
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .split('-')
                .last()
                .unwrap()
                .parse::<usize>()
                .unwrap();

            let m = get_all_matches(x.clone(), &progress, rep, seed);
            write_match_data(&m, exp_output_path.join(format!("matches-{}.csv", rep))).unwrap()
        })
        .collect::<Vec<_>>();

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        println!("{}", err);
        process::exit(1);
    }
}

// main function that will be run in a single thread for each replicate
fn get_all_matches(
    data: path::PathBuf,
    progress: &Mutex<Progress>,
    rep: usize,
    seed: u64,
) -> Vec<Match> {
    // read the imputed data with polars
    let d = read_imputed_table(data.join("base_strs_imputed.tsv")).unwrap();
    let d_true = read_true_table(data.join("base_strs.tsv")).unwrap();
    let samples = convert_data(d, d_true).unwrap();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = vec![];

    // make all distinct pairs i,j of samples
    let pairs = (0..samples.len())
        .flat_map(|i| (i + 1..samples.len()).map(move |j| (i, j)))
        .collect::<Vec<_>>();

    let bar: Bar = progress
        .lock()
        .unwrap()
        .bar(pairs.len(), format!("Working on {}:", rep));

    for (cnt, (i, j)) in pairs.into_iter().enumerate() {
        let m = pair_matches(&samples, i, j, rep, &mut rng);
        result.push(m);

        if cnt % 1000 == 0 {
            progress.lock().unwrap().inc_and_draw(&bar, 1000);
        }
    }

    result
}

fn compare_genotypes_full(a: (usize, usize), b: (usize, usize)) -> bool {
    (a.0 == b.0 && a.1 == b.1) || (a.0 == b.1 && a.1 == b.0)
}

fn compare_genotypes_partial(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 == b.0 || a.0 == b.1 || a.1 == b.0 || a.1 == b.1
}

fn pair_matches(samples: &[Sample], i: usize, j: usize, rep: usize, rng: &mut impl Rng) -> Match {
    let (fs, ps, ns, als) = simulate_expected_matches(samples, i, j, rng);
    let sim_full_matches = fs.iter().sum::<f64>() / NSIM as f64;
    let sim_partial_matches = ps.iter().sum::<f64>() / NSIM as f64;
    let sim_no_matches = ns.iter().sum::<f64>() / NSIM as f64;
    let sim_allele_matches = als.iter().sum::<f64>() / NSIM as f64;

    let (mean_full_matches, mean_partial_matches, mean_no_matches, mean_allele_matches) =
        compute_expected_matches(samples, i, j);

    // calculate called matches using the imp_gt value for each sample
    // calculate true matches using the true_gt value for each sample
    let mut called_fs: usize = 0;
    let mut called_ps: usize = 0;
    let mut called_ns: usize = 0;

    let mut true_fs: usize = 0;
    let mut true_ps: usize = 0;
    let mut true_ns: usize = 0;

    for k in 0..samples[i].imp_gt.len() {
        let g1i = samples[i].imp_gt[k];
        let g2i = samples[j].imp_gt[k];
        let g1t = samples[i].true_gt[k];
        let g2t = samples[j].true_gt[k];

        if compare_genotypes_full(g1i, g2i) {
            called_fs += 1;
        } else if compare_genotypes_partial(g1i, g2i) {
            called_ps += 1;
        } else {
            called_ns += 1;
        }

        if compare_genotypes_full(g1t, g2t) {
            true_fs += 1;
        } else if compare_genotypes_partial(g1t, g2t) {
            true_ps += 1;
        } else {
            true_ns += 1;
        }
    }

    // average the numbers over all simulated comparisons
    Match {
        rep,
        s1: i,
        s2: j,
        mean_full_matches,
        mean_partial_matches,
        mean_no_matches,
        mean_allele_matches,
        sim_full_matches,
        sim_partial_matches,
        sim_no_matches,
        sim_allele_matches,
        called_full_matches: called_fs,
        called_partial_matches: called_ps,
        called_no_matches: called_ns,
        called_allele_matches: 2 * called_fs + called_ps,
        true_full_matches: true_fs,
        true_partial_matches: true_ps,
        true_no_matches: true_ns,
        true_allele_matches: 2 * true_fs + true_ps,
    }
}

fn compute_expected_matches(samples: &[Sample], i: usize, j: usize) -> (f64, f64, f64, f64) {
    let nloci = samples[i].nloci;

    let mut pi = vec![vec![vec![0.0; nloci]; nloci]; nloci];

    pi[0][0][0] = p00(
        &samples[i].ap1[0],
        &samples[i].ap2[0],
        &samples[j].ap1[0],
        &samples[j].ap2[0],
    );
    pi[0][0][1] = p01(
        &samples[i].ap1[0],
        &samples[i].ap2[0],
        &samples[j].ap1[0],
        &samples[j].ap2[0],
    );
    pi[0][1][0] = p10(
        &samples[i].ap1[0],
        &samples[i].ap2[0],
        &samples[j].ap1[0],
        &samples[j].ap2[0],
    );

    // now do dynamic programming on this
    for l in 1..nloci {
        for m in 0..nloci {
            for p in 0..nloci {
                if m + p > l {
                    continue;
                } else if m == 0 && p == 0 {
                    pi[l][m][p] = pi[l - 1][m][p]
                        * p00(
                            &samples[i].ap1[l - 1],
                            &samples[i].ap2[l - 1],
                            &samples[j].ap1[l - 1],
                            &samples[j].ap2[l - 1],
                        )
                } else if m == 0 {
                    pi[l][m][p] = pi[l - 1][m][p]
                        * p00(
                            &samples[i].ap1[l - 1],
                            &samples[i].ap2[l - 1],
                            &samples[j].ap1[l - 1],
                            &samples[j].ap2[l - 1],
                        )
                        + pi[l - 1][m][p - 1]
                            * p01(
                                &samples[i].ap1[l - 1],
                                &samples[i].ap2[l - 1],
                                &samples[j].ap1[l - 1],
                                &samples[j].ap2[l - 1],
                            )
                } else if p == 0 {
                    pi[l][m][p] = pi[l - 1][m - 1][p]
                        * p10(
                            &samples[i].ap1[l - 1],
                            &samples[i].ap2[l - 1],
                            &samples[j].ap1[l - 1],
                            &samples[j].ap2[l - 1],
                        )
                        + pi[l - 1][m][p]
                            * p00(
                                &samples[i].ap1[l - 1],
                                &samples[i].ap2[l - 1],
                                &samples[j].ap1[l - 1],
                                &samples[j].ap2[l - 1],
                            )
                } else {
                    pi[l][m][p] = pi[l - 1][m - 1][p]
                        * p10(
                            &samples[i].ap1[l - 1],
                            &samples[i].ap2[l - 1],
                            &samples[j].ap1[l - 1],
                            &samples[j].ap2[l - 1],
                        )
                        + pi[l - 1][m][p - 1]
                            * p01(
                                &samples[i].ap1[l - 1],
                                &samples[i].ap2[l - 1],
                                &samples[j].ap1[l - 1],
                                &samples[j].ap2[l - 1],
                            )
                        + pi[l - 1][m][p]
                            * p00(
                                &samples[i].ap1[l - 1],
                                &samples[i].ap2[l - 1],
                                &samples[j].ap1[l - 1],
                                &samples[j].ap2[l - 1],
                            )
                }
            }
        }
    }

    // now compute expectations
    // redefine pi as the last element of pi
    let pi = pi[nloci - 1].clone();

    let mut full_expectation = 0.0;
    let mut partial_expectation = 0.0;
    let mut no_expectation = 0.0;
    let mut allele_expectation = 0.0;
    for (m, row) in pi.iter().enumerate() {
        for (p, prob) in row.iter().enumerate() {
            full_expectation += prob * (m + 1) as f64;
            partial_expectation += prob * (p + 1) as f64;
            no_expectation += prob * (nloci - m - p - 2) as f64;
            allele_expectation += prob * (2 * (m + 1) + (p + 1)) as f64;
        }
    }

    (
        full_expectation,
        partial_expectation,
        no_expectation,
        allele_expectation,
    )
}

fn p10(a1: &[f64], a2: &[f64], b1: &[f64], b2: &[f64]) -> f64 {
    let nl = a1.len();
    let mut p = 0.0;

    for i in 0..nl {
        for j in 0..nl {
            if i != j {
                p += a1[i] * a2[j] * (b1[i] * b2[j] + b1[j] * b2[i]);
            }
        }

        p += a1[i] * a2[i] * b1[i] * b2[i];
    }

    p
}

fn p01(a1: &[f64], a2: &[f64], b1: &[f64], b2: &[f64]) -> f64 {
    let nl = a1.len();
    let mut p = 0.0;

    for i in 0..nl {
        for j in 0..nl {
            if i != j {
                p += a1[i]
                    * a2[j]
                    * (b1[i] * (1. - b2[j])
                        + b1[j] * (1. - b2[i])
                        + (1. - b1[i] - b1[j]) * (b2[i] + b2[j]));
            }
        }

        p += a1[i] * a2[i] * (b1[i] * (1. - b2[i]) + (1. - b1[i]) * b2[i]);
    }

    p
}

fn p00(a1: &[f64], a2: &[f64], b1: &[f64], b2: &[f64]) -> f64 {
    let nl = a1.len();
    let mut p = 0.0;

    for i in 0..nl {
        for j in 0..nl {
            if i != j {
                p += a1[i] * a2[j] * (1. - b1[i] - b1[j]) * (1. - b2[i] - b2[j]);
            }
        }

        p += a1[i] * a2[i] * (1. - b1[i]) * (1. - b2[i]);
    }

    p
}

fn simulate_expected_matches(
    samples: &[Sample],
    i: usize,
    j: usize,
    rng: &mut impl Rng,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // number of full matches in each simulated comparison
    let mut fs = vec![0.0; NSIM];
    // number of partial matches in each simulated comparison
    let mut ps = vec![0.0; NSIM];
    // number of no matches in each simulated comparison
    let mut ns = vec![0.0; NSIM];
    // number of allele matches in each simulated comparison
    let mut als = vec![0.0; NSIM];

    // do many simulated comparisons
    for r in 0..NSIM {
        // sample genotypes for each sample
        let s1 = sample_genotype(&samples[i], rng);
        let s2 = sample_genotype(&samples[j], rng);

        // for each locus, check if the genotypes match
        for k in 0..s1.len() {
            let g1 = s1[k];
            let g2 = s2[k];

            if compare_genotypes_full(g1, g2) {
                fs[r] += 1.0;
            } else if compare_genotypes_partial(g1, g2) {
                // very important: partial match is conditional on no full match!!
                ps[r] += 1.0;
            } else {
                ns[r] += 1.0;
            }
        }

        als[r] = 2. * fs[r] + ps[r];
    }
    (fs, ps, ns, als)
}

fn sample_genotype(s: &Sample, rng: &mut impl Rng) -> Vec<(usize, usize)> {
    let mut result = vec![(0, 0); s.nloci];

    for (i, r) in result.iter_mut().enumerate() {
        let allele1 = sample_allele(&s.ap1[i], rng);
        let allele2 = sample_allele(&s.ap2[i], rng);

        *r = (allele1, allele2);
    }

    result
}

fn sample_allele(p: &[f64], rng: &mut impl Rng) -> usize {
    let mut r = rng.gen::<f64>();
    let mut i = 0;

    while r > p[i] {
        r -= p[i];
        i += 1;
    }

    i
}

// reading in the tsv,,, weird polars code incoming
fn read_imputed_table(data: path::PathBuf) -> Result<DataFrame, Box<dyn Error>> {
    // define the schema with all string fields
    let sch = Arc::new(Schema::from(
        vec![
            Field::new("locus", DataType::Utf8),
            Field::new("sample", DataType::Utf8),
            Field::new("genotype", DataType::Utf8),
            Field::new("ds", DataType::Utf8),
            Field::new("ap1", DataType::Utf8),
            Field::new("ap2", DataType::Utf8),
            Field::new("gp", DataType::Utf8),
        ]
        .into_iter(),
    ));

    // read in the tsv
    let mut d = CsvReader::from_path(data)?
        .with_delimiter(b'\t')
        .has_header(false)
        .with_schema(sch)
        .finish()?;

    let descending = vec![false, false];
    let by = &["sample", "locus"];

    // sort and subset
    d = d
        .select(["sample", "locus", "genotype", "ap1", "ap2"])?
        .sort(by, descending)?;

    Ok(d)
}

fn read_true_table(data: path::PathBuf) -> Result<DataFrame, Box<dyn Error>> {
    // define the schema with all string fields
    let sch = Arc::new(Schema::from(
        vec![
            Field::new("locus", DataType::Utf8),
            Field::new("sample", DataType::Utf8),
            Field::new("genotype", DataType::Utf8),
        ]
        .into_iter(),
    ));

    // read in the tsv
    let mut d = CsvReader::from_path(data)?
        .with_delimiter(b'\t')
        .has_header(false)
        .with_schema(sch)
        .finish()?;

    let descending = vec![false, false];
    let by = &["sample", "locus"];

    // sort and subset
    d = d.sort(by, descending)?;

    Ok(d)
}

// convert the data into a vector of simple structs (in python this was a dataframe of ndarrays, very clumsy)
// should be better here
fn convert_data(d: DataFrame, d_true: DataFrame) -> Result<Vec<Sample>, Box<dyn Error>> {
    // get sample names
    let samples = d.column("sample")?.unique()?;
    let samples = samples.utf8()?.into_no_null_iter().collect::<Vec<_>>();

    // verify the number of loci again
    let loci = d.column("locus")?.unique()?;
    let loci = loci.utf8()?.into_no_null_iter().collect::<Vec<_>>();
    let nloci = loci.len();

    let mut result: Vec<Sample> = vec![];

    // for each sample id, extract the allele probabilities
    for s in samples {
        // make a column for filtering and subset d
        let idx = d.column("sample")?.equal(s)?;
        let ds = d.filter(&idx)?;
        let ds_true = d_true.filter(&idx)?;

        // extract columns as some internal polars type
        let ap1 = ds.column("ap1")?.utf8()?;
        let ap2 = ds.column("ap2")?.utf8()?;
        let imp_gt = ds.column("genotype")?.utf8()?;
        let true_gt = ds_true.column("genotype")?.utf8()?;

        // convert to iterators and process so that probabilities add to 1
        let ap1 = ap1
            .into_no_null_iter()
            .map(process_probabilities)
            .collect::<Vec<_>>();
        let ap2 = ap2
            .into_no_null_iter()
            .map(process_probabilities)
            .collect::<Vec<_>>();
        let imp_gt = imp_gt
            .into_no_null_iter()
            .map(process_genotype)
            .collect::<Vec<_>>();
        let true_gt = true_gt
            .into_no_null_iter()
            .map(process_genotype)
            .collect::<Vec<_>>();

        assert_eq!(ap1.len(), nloci);
        assert_eq!(ap2.len(), nloci);
        assert_eq!(imp_gt.len(), nloci);
        assert_eq!(true_gt.len(), nloci);
        assert!(ap1.iter().zip(ap2.iter()).all(|(x, y)| x.len() == y.len()));

        // conbine into a struct
        result.push(Sample {
            nloci,
            true_gt,
            imp_gt,
            ap1,
            ap2,
        });
    }

    Ok(result)
}

// function process_genotype receives a string "12|5" and returns a tuple of usize (12, 5)
fn process_genotype(x: &str) -> (usize, usize) {
    let mut y = x.split('|');
    let a = y.next().unwrap().parse::<usize>().unwrap();
    let b = y.next().unwrap().parse::<usize>().unwrap();

    (a, b)
}

// makes the string "0.2,0.42,0.1" into an f64 vector, and adds the probability of base (non-alternate) allele to the front
fn process_probabilities(x: &str) -> Vec<f64> {
    let mut result = x
        .split(',')
        .map(|y| y.parse::<f64>().unwrap())
        .collect::<Vec<f64>>();

    let s = result.iter().sum::<f64>();
    result.insert(0, (1.0 - s).max(0.0));
    result
}

fn write_match_data(m: &Vec<Match>, output: path::PathBuf) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(output)?;

    for m in m {
        wtr.serialize(m)?;
    }

    wtr.flush()?;

    Ok(())
}

#[derive(Debug, Clone)]
struct Sample {
    nloci: usize,
    true_gt: Vec<(usize, usize)>,
    imp_gt: Vec<(usize, usize)>,
    ap1: Vec<Vec<f64>>,
    ap2: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct Match {
    rep: usize,
    s1: usize,
    s2: usize,
    mean_full_matches: f64,
    mean_partial_matches: f64,
    mean_no_matches: f64,
    mean_allele_matches: f64,
    sim_full_matches: f64,
    sim_partial_matches: f64,
    sim_no_matches: f64,
    sim_allele_matches: f64,
    called_full_matches: usize,
    called_partial_matches: usize,
    called_no_matches: usize,
    called_allele_matches: usize,
    true_full_matches: usize,
    true_partial_matches: usize,
    true_no_matches: usize,
    true_allele_matches: usize,
}
