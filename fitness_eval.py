from flatspin.data import Dataset

def worker_main(args):
    dataset = Dataset.read(args.basepath)

    worker_id = args.worker_id
    num_jobs = len(dataset)
    num_workers = args.num_workers

    # Calculate which jobs in the dataset to run
    from_idx = (worker_id * num_jobs) // num_workers
    to_idx = ((worker_id + 1) * num_jobs) // num_workers

    dataset = dataset[from_idx:to_idx-1]
    #run_local(dataset)



if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(description= 'Run a fitness function on a dataset')
    parser.add_argument('-o', '--basepath', help='output directory for results')
    parser.add_argument('--worker-id', help='worker id', type=int)
    parser.add_argument('--num-workers', help='number of workers in total', type=int)

    args = parser.parse_args()
    worker_main(args)
