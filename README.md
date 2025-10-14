### Local development
To start model training:
- Select model in `program.py` or determine configuration
- Run command `python3 program.py train -p H:\Projects\University\NeauralNetworks\train_2025_01_05x256 -d mnist256 --first_layers 3,5,7,9,11 --second_layers 3,5,7,9,11 --iterate True --dataset_path H:\Projects\University\NeauralNetworks\Datasets\mnist_scaled_256`

Using `config.json` file you can determine multiple model consfigurations and run them sequentially
`python3 program.py train --config <path-to-config>`

To test model:
- Select model in `program.py`
- Run command `python3 program.py performance -p H:\Projects\University\NeauralNetworks\train_2025_01_05x256 -d mnist256 --first_layers 3,5,7,9,11 --second_layers 3,5,7,9,11 --iterate True --dataset_path H:\Projects\University\NeauralNetworks\Datasets\mnist_scaled_256`

To visualize results:
- `python3 accuracy_per_layer_visualization.py show --path H:\Projects\University\NeauralNetworks\train_2025_01_05_cifar10x128`

Scale dataset images:
- `python3 scaler.py --dataset cifar10 --target_size 128 --path <path>`