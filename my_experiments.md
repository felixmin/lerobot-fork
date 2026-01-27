SmolLVA

- Works at the moment with the following start command
python src/lerobot/scripts/lerobot_train.py   --dataset.repo_id lerobot/pusht   --dataset.video_backend pyav   --env.type pusht   --policy.type smolvla   --policy.device cuda   --policy.use_amp true   --policy.push_to_hub false   --wandb.enable true   --wandb.project "$WANDB_PROJECT"   --wandb.entity "$WANDB_ENTITY"   --wandb.mode online   --wandb.disable_artifact true   --job_name "$WANDB_NAME"   --output_dir "outputs/$WANDB_NAME"   --steps 2000   --log_freq 10   --eval_freq 200   --eval.n_episodes 20   --eval.batch_size 10   --save_checkpoint true   --save_freq 200   --batch_size 64   --num_workers 8


Pi0.5




VLA-0

- Thats another repo look at .../code/vla0





lerobot-train 
--job_name=smolvla_libero100  
--output_dir=outputs/train/smolvla_libero16 

--policy.type=smolvla 
--policy.push_to_hub=false 
--policy.load_vlm_weights=true 
--policy.n_action_steps=50 ???
# --policy.chunk_size=50 ???
# --policy.train_expert_only=true
# --policy.freeze_vision_encoder=true

--dataset.repo_id=HuggingFaceVLA/libero # or use lerobot/libero ?  
--dataset.use_imagenet_stats=false 
--dataset.video_backend=pyav 
# --dataset.use_imagenet_stats=false

--env.type=libero 
--env.task=libero_10

--eval_freq=10000 
--eval.n_episodes=6 
--eval.batch_size=2 

--wandb.enable=true 

--batch_size=64 
--steps=200000 
--save_freq=20000 





