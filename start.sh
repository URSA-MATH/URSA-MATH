pip3 uninstall -y vllm
export VLLM_COMMIT=0b8bb86bf19d68950b4d92a99350e07a26ec0d2c # use full commit hash from the main branch
pip3 install https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
cd ./vllm/vllm
mkdir vllm_flash_attn
cd ..
python3 python_only_dev.py