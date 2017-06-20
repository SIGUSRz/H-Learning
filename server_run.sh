srun -n1 -N1 -p serial --mem=4000 --gres=gpu:1 --pty bash
./run_gpu.sh
