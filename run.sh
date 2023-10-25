for command in delete_incomplete launch
do
  python -m domainbed.scripts.sweep ${command} --data_dir=./data \
  --output_dir=./output --command_launcher "local" --algorithms ERM_SPSDViT \
  --single_test_envs  --datasets DR --n_hparams 1 --n_trials 3  \
  --hparams """{\"backbone\":\"CVTSmall\",\"batch_size\":32,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0}"""
done
