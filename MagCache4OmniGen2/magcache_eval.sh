# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="OmniGen2/OmniGen2"

# image edit in example_edit.sh
python inference.py \
--model_path $model_path \
--num_inference_step 50 \
--text_guidance_scale 5.0 \
--image_guidance_scale 2.0 \
--instruction "Change the background to classroom." \
--input_image_path example_images/ComfyUI_temp_mllvz_00071_.png \
--output_image_path outputs/output_edit_magcache_002.png \
--num_images_per_prompt 1 \
--enable_magcache \
--magcache_thresh 0.02

# image edit in example_edit_test.sh
python inference.py \
--model_path $model_path \
--num_inference_step 30 \
--text_guidance_scale 5.0 \
--image_guidance_scale 2.0 \
--instruction "Change the color of dress to light green." \
--input_image_path example_images/flux5.png \
--output_image_path outputs/prompt_guide_edit_1_magcache_002.png \
--num_images_per_prompt 1 \
--cfg_range_end 0.8 \
--scheduler dpmsolver++ \
--enable_magcache \
--magcache_thresh 0.02

# in context generation in example_in_context_generation.sh
python inference.py \
--model_path $model_path \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 5.0 \
--image_guidance_scale 2.0 \
--instruction "Please let the person in image 2 hold the toy from the first image in a parking lot." \
--input_image_path example_images/04.jpg example_images/000365954.jpg \
--output_image_path outputs/output_in_context_generation_magcache_002.png \
--num_images_per_prompt 1 \
--enable_magcache \
--magcache_thresh 0.02

# text to image generation in example_t2i.sh
python inference.py \
--model_path $model_path \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 4.0 \
--instruction "The sun rises slightly, the dew on the rose petals in the garden is clear, a crystal ladybug is crawling to the dew, the background is the early morning garden, macro lens." \
--output_image_path outputs/output_t2i_magcache_001.png \
--num_images_per_prompt 1 \
--enable_magcache \
--magcache_thresh 0.01
