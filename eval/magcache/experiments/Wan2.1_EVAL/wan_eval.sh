mkdir ./logs
# MagCache

## magcache-slow
total_samples=950 # 950 for the Final Performance, 100 for ablation
num_gpus=8
per_gpu=$((total_samples / num_gpus))  # number of samples per card

for ((i=0; i<num_gpus; i++)); do
    # compute start and end index
    start=$((i * per_gpu))
    end=$(( (i + 1) * per_gpu ))
    
    if (( i == num_gpus - 1 )); then
        end=$total_samples
    fi

    (
        export CUDA_VISIBLE_DEVICES=$i
        python wan_magcache.py \
            --task t2v-1.3B \
            --size 832*480 \
            --sample_shift 8 \
            --sample_guide_scale 6 \
            --sample_steps 50 \
            --ckpt_dir /data/data/pretrained_models/Wan2.1-T2V-1.3B \
            --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." \
            --base_seed 0 \
            --enable_magcache \
            --magcache_thresh 0.12 \
            --magcache_K 2 \
            --save_dir ./generated_samples/wan_magcache_slow/ \
            --start_index $start \
            --end_index $end  > "./logs/magcache_slow_012_max2steps_gpu${i}.log" 2>&1
    ) & 
done
wait  # 等待所有后台任务
echo "All ${num_gpus} GPUs have completed processing!"

## magcache-fast
total_samples=950 # 950 for the Final Performance, 100 for ablation
num_gpus=8
per_gpu=$((total_samples / num_gpus))  # number of samples per card

for ((i=0; i<num_gpus; i++)); do
    # compute start and end index
    start=$((i * per_gpu))
    end=$(( (i + 1) * per_gpu ))
    
    if (( i == num_gpus - 1 )); then
        end=$total_samples
    fi

    (
        export CUDA_VISIBLE_DEVICES=$i
        python wan_magcache.py \
            --task t2v-1.3B \
            --size 832*480 \
            --sample_shift 8 \
            --sample_guide_scale 6 \
            --sample_steps 50 \
            --ckpt_dir /data/data/pretrained_models/Wan2.1-T2V-1.3B \
            --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." \
            --base_seed 0 \
            --enable_magcache \
            --magcache_thresh 0.12 \
            --magcache_K 4 \
            --save_dir ./generated_samples/wan_magcache_fast/ \
            --start_index $start \
            --end_index $end  > "./logs/magcache_fast_012_max4steps_gpu${i}.log" 2>&1
    ) & 
done
wait  # 等待所有后台任务
echo "All ${num_gpus} GPUs have completed processing!"


# generate original videos
total_samples=950 # 950 for the Final Performance, 100 for ablation
num_gpus=8
per_gpu=$((total_samples / num_gpus))  # number of samples per card

for ((i=0; i<num_gpus; i++)); do
    # compute start and end index
    start=$((i * per_gpu))
    end=$(( (i + 1) * per_gpu ))
    
    if (( i == num_gpus - 1 )); then
        end=$total_samples
    fi

    (
        export CUDA_VISIBLE_DEVICES=$i
        python wan_magcache.py \
            --task t2v-1.3B \
            --size 832*480 \
            --sample_shift 8 \
            --sample_guide_scale 6 \
            --sample_steps 50 \
            --ckpt_dir /data/data/pretrained_models/Wan2.1-T2V-1.3B \
            --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." \
            --base_seed 0 \
            --save_dir ./generated_samples/wan_original/ \
            --start_index $start \
            --end_index $end  > "./logs/wan_original_gpu${i}.log" 2>&1
    ) & 
done
wait  # 等待所有后台任务
echo "All ${num_gpus} GPUs have completed processing!"


# generate video with teacache

## teacache-slow
total_samples=950
num_gpus=8
per_gpu=$((total_samples / num_gpus))

for ((i=0; i<num_gpus; i++)); do
    start=$((i * per_gpu))
    end=$(( (i + 1) * per_gpu ))

    if (( i == num_gpus - 1 )); then
        end=$total_samples
    fi

    (
        export CUDA_VISIBLE_DEVICES=$i
        python wan_teacache.py \
            --task t2v-1.3B \
            --size 832*480 \
            --sample_shift 8 \
            --sample_guide_scale 6 \
            --sample_steps 50 \
            --ckpt_dir /data/data/pretrained_models/Wan2.1-T2V-1.3B \
            --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." \
            --base_seed 0 \
            --save_dir ./generated_samples/wan_teacache_slow/ \
            --teacache_thresh 0.05 \
            --start_index $start \
            --end_index $end > "./logs/teacache_slow_005_gpu${i}.log" 2>&1
    ) &  # 后台执行
done

wait  # 等待所有后台任务
echo "All ${num_gpus} GPUs have completed processing!"

## teacache-fast
total_samples=950
num_gpus=8
per_gpu=$((total_samples / num_gpus))

for ((i=0; i<num_gpus; i++)); do
    start=$((i * per_gpu))
    end=$(( (i + 1) * per_gpu ))

    if (( i == num_gpus - 1 )); then
        end=$total_samples
    fi

    (
        export CUDA_VISIBLE_DEVICES=$i
        python wan_teacache.py \
            --task t2v-1.3B \
            --size 832*480 \
            --sample_shift 8 \
            --sample_guide_scale 6 \
            --sample_steps 50 \
            --ckpt_dir /data/data/pretrained_models/Wan2.1-T2V-1.3B \
            --prompt "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." \
            --base_seed 0 \
            --save_dir ./generated_samples/wan_teacache_fast/ \
            --teacache_thresh 0.08 \
            --start_index $start \
            --end_index $end > "./logs/teacache_fast_008_gpu${i}.log" 2>&1
    ) &  # 后台执行
done

wait  # 等待所有后台任务
echo "All ${num_gpus} GPUs have completed processing!"