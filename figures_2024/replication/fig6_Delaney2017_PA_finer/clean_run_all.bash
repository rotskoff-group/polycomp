for D in $(find . -name 'gibbs*'); do
    if [ -d "${D}" ]; then
        cd "${D}"
        rm slurm-*
        rm *.npy
        sed -i '2s/.*/sys.path.insert(0, "\/scratch\/users\/epert\/polycomp\/gpu_polycomp")/' coacervate_sherlock.py
        sed -i '2s/.*/#/' run_gibbs.slurm
        sed -i '5s/.*/#SBATCH --time=30:00:00/' run_gibbs.slurm
        sbatch run_gibbs.slurm
        cd ..
    fi
done

