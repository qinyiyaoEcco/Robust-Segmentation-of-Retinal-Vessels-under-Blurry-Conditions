## Usage

To train the model with both channel attention and multi-task learning:

```bash
python train.py --data_dir ./data --batch_size 4 --epochs 50 --lr 0.001 --save_dir ./results --use_attention --multitask
```

Parameters:
- `--data_dir`: Path to the dataset directory
- `--batch_size`: Batch size (default: 4)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--save_dir`: Directory to save models and logs (default: ./results)
- `--use_attention`: Enable channel attention mechanism
- `--multitask`: Enable multi-task learning

You can selectively enable features:
- For only attention: `--use_attention`
- For only multi-task learning: `--multitask`
- For both: `--use_attention --multitask`
- For baseline model: (use neither flag)

### Evaluation

To evaluate the model:

```bash
python evaluate.py --data_dir ./data --model_path ./results/best_model.pth --output_dir ./predictions --use_attention --multitask
```

这个代码默认会开启通道注意力

Parameters:

- `--data_dir`: Path to the dataset directory
- `--model_path`: Path to the trained model file (required)
- `--output_dir`: Directory to save prediction results (default: ./predictions)
- `--batch_size`: Batch size (default: 1)
- `--use_attention`: Enable channel attention mechanism (must match training setting)
