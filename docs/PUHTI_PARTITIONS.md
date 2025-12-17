Available batch job partitions
On CSC supercomputers, programs are run by submitting them to partitions, which are logical sets of nodes managed by the SLURM workload manager. This page lists the available SLURM partitions on the Puhti and Mahti supercomputers, as well as explains their intended uses. Below are the general guidelines for using the SLURM partitions on our systems:

Use the test and gputest partitions for testing your code, not production. These partitions provide access to fewer resources than other partitions, but jobs submitted to them have a higher priority and are thus granted resources before other jobs.
Only request multiple CPU cores if you know your program supports parallel processing. Reserving multiple cores does not automatically speed up your job. Your program must be written in a way that the computations can be done in multiple threads or processes. Reserving more cores does nothing by itself, except make you queue for longer.
Only use the GPU partitions if you know your program can utilize GPUs. Running your computations using one or more GPUs is a very effective parallelization method for certain applications, but your program must be configured to use the CUDA platform. If you are unsure whether this is the case, it is better to submit it to a CPU partition, since you will be allocated resources sooner. You may also always consult CSC Service Desk when in doubt.
The following commands can be used to show information about available partitions:


# Display a summary of available partitions
$ sinfo --summarize

# Display details about a specific partition:
$ scontrol show partition <partition_name>
LUMI partitions

The available LUMI batch job partitions are found in the LUMI documentation.

Puhti partitions
The following guidelines apply to the SLURM partitions on Puhti:

Only request the memory you need. Memory can easily end up being a bottleneck in resource allocation. Even if the desired amount of GPUs and/or CPU cores is continuously available, your job will sit in the queue for as long as it takes for the requested amount of memory to become free. It is thus recommended to only request the amount of memory that is necessary for running your job. Additionally, the amount of CPU/GPU Billing Units consumed by your job is affected by the amount of memory requested, not the amount which was actually used. See how to estimate your memory requirements.
Only use the longrun partitions if necessary. The longrun and hugemem_longrun partitions provide access to fewer resources and have a lower priority than the other partitions, so it is recommended to use them only for jobs that really require a very long runtime (e.g. if there is no way to checkpoint and restart a computation).
Puhti CPU partitions
Puhti features the following partitions for submitting jobs to CPU nodes:

Partition	Time
limit	Max CPU
cores	Max
nodes	Node types	Max memory
per node	Max local storage
(NVMe) per node
test	15 minutes	80	2	M	185 GiB	n/a
small	3 days	40	1	M, L, IO	373 GiB	3600 GiB
large	3 days	1040	26	M, L, IO	373 GiB	3600 GiB
longrun	14 days	40	1	M, L, IO	373 GiB	3600 GiB
hugemem	3 days	160	4	XL, BM	1496 GiB	1490 GiB (XL), 5960 GiB (BM)
hugemem_longrun	14 days	40	1	XL, BM	1496 GiB	1490 GiB (XL), 5960 GiB (BM)
Puhti GPU partitions
Puhti features the following partitions for submitting jobs to GPU nodes:

Partition	Time
limit	Max
GPUs	Max CPU
cores	Max
nodes	Node types	Max memory
per node	Max local storage
(NVMe) per node
gputest	15 minutes	8	80	2	GPU	373 GiB	3600 GiB
gpu	3 days	80	800	20	GPU	373 GiB	3600 GiB
Fair use of GPU nodes on Puhti

You should reserve no more than 10 CPU cores per GPU.

Puhti interactive partition
The interactive partition on Puhti allows running interactive jobs on CPU nodes. To run an interactive job on a GPU node, use sinteractive command with the -g option, which submits the job to the gpu partition instead. Note that you can only run two simultaneous jobs on the Puhti interactive partition.

Partition	Time
limit	Max CPU
cores	Max
nodes	Node types	Max memory
per node	Max local storage
(NVMe) per node
interactive	7 days	8	1	IO	76 GiB	720 GiB