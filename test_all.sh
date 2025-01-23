

# 0.00, 0.05, 0.10, 0.15, 0.20
noises=("0.00" "0.05" "0.10" "0.15" "0.20" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5" "0.55")
# noises_all=("0.000" "0.001" "0.002" "0.004" "0.008" "0.016" "0.032" "0.064" "0.128" "0.256" "0.512" "1.024")
# num_noise_per_machine=2
# let start_index=($1-1)*$num_noise_per_machine
# noises=("${noises_all[@]:$start_index:$num_noise_per_machine}")
# echo "Selected noises: (${noises[*]})"
# # exit
# noises=("0.000" "0.001" "0.002" "0.004" "0.008" "0.016" "0.032" "0.064" "0.128" "0.256" "0.512" "1.024") 
# noises=("0.000" "0.001" "0.002" "0.004" "0.008" "0.016")
# noises=("0.032" "0.064" "0.128" "0.256" "0.512" "1.024")
# noises=("0.000" "0.001" "0.002") 
# noises=("0.004" "0.008" "0.016" ) 
# noises=("0.032" "0.064" "0.128") 
# noises=("0.256" "0.512" "1.024") 
# freq=34

basis_arr=( 50 )
save_dir=results_test
mkdir -p ${save_dir}

for noise in "${noises[@]}"
do
    for dataset in "default_0" #"default_10" "default_11" "default_12" "default_13"
    do        
        for env in $(seq 0 4)
        do
            for ode in "FrictionPendulum" #"Lorenz" #"LvODE" "SirODE" 
            do
                if [[ $ode == "Lorenz" ]]; then
                    x_id_list=(0 1 2)
                    freq=34
                elif [[ $ode == "SirODE" ]]; then
                    x_id_list=(0 1 2)
                    freq=4.16666667
                elif [[ $ode == "FrictionPendulum" ]]; then
                    x_id_list=(0 1)
                    freq=100
                else
                    x_id_list=(0 1)
                    freq=25
                fi
                for seed in $1
                do
                    for x_id in "${x_id_list[@]}"
                    do
                        for n_basis in "${basis_arr[@]}"
                        do
                            mkdir -p ${save_dir}/${ode}/${dataset}
                            echo "python -u run_sensitivity_vi.py --ode_name=${ode} --basis=sine --n_basis=${n_basis} --seed=${seed} --freq=${freq}  --noise_ratio=${noise} --x_id=${x_id} --n_sample=1 --env=${env} --dataset=${dataset} --save_dir=${save_dir} 2>&1 | tee -a ${save_dir}/noise-${noise}-seed-${seed}-env-${env}.txt"
                            python -u run_sensitivity_vi.py --ode_name=${ode} --basis=sine --n_basis=${n_basis} --seed=${seed} --freq=${freq}  --noise_ratio=${noise} --x_id=${x_id} --n_sample=1 --env=${env} --dataset=${dataset} --save_dir=${save_dir} 2>&1 | tee -a ${save_dir}/noise-${noise}-seed-${seed}-env-${env}.txt
                            # exit
                            # sleep 1
                        done
                    done
                done
            done
        done
    done
done


# alg_array=( sine )
# rm results/sensitivity_${ode}.txt

# for n_basis in "${basis_arr[@]}"
# do
#     for basis in "${alg_array[@]}"
#     do
#         python -u evaluation_sensitivity.py --ode_name=${ode} --basis=${basis} --n_basis=${n_basis} --seed=${seed} --freq=${freq}  --noise_sigma=${noise} --x_id=${x_id} --n_seed=${n_seed} --n_sample=50 >> results/sensitivity_${ode}.txt
#     done
# done

# cat results/sensitivity_${ode}.txt

