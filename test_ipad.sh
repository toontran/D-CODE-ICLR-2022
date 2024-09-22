
n_basis=50

save_dir=results_test
mkdir -p ${save_dir}
test_folder=data.pkl

for env in $(seq 0 4) 
do
    echo "python -u run_sensitivity_vi.py --basis=sine --n_basis=${n_basis} --n_sample=1 --env=${env} --save_dir=${save_dir} --load_ipad_data=${test_folder} 2>&1 | tee -a ${save_dir}/ipad.txt"
    python -u run_sensitivity_vi.py --basis=sine --n_basis=${n_basis} --n_sample=1 --env=${env} --save_dir=${save_dir} --load_ipad_data=${test_folder} 2>&1 | tee -a ${save_dir}/ipad.txt
    # exit 
done


