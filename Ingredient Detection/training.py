from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

if __name__ == '__main__':

    model = YOLO('yolov8m.pt')

    results = model.train(
        data='ingredients_data.yaml',
        epochs=100,  
        imgsz=640,
        batch=16,
        device=0,
        patience=20,  
        project='runs/ingredient_detection',
        name='demo',
        verbose=True,  
        plots=True     
    )
    
    print("\n" + "="*50)
    print("GENERATING TRAINING PLOTS...")
    print("="*50 + "\n")
    
    # Get the results directory
    results_dir = Path('runs/ingredient_detection/demo')
    csv_path = results_dir / 'results.csv'
    
    if csv_path.exists():
        # Load training results
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # Clean column names
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Results Summary', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        if 'train/box_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
        if 'train/cls_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', linewidth=2)
        if 'train/dfl_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Losses Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        if 'metrics/precision(B)' in df.columns:
            ax2.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, marker='o', markersize=3)
        if 'metrics/recall(B)' in df.columns:
            ax2.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, marker='s', markersize=3)
        if 'metrics/mAP50(B)' in df.columns:
            ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2, marker='^', markersize=3)
        if 'metrics/mAP50-95(B)' in df.columns:
            ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2, marker='d', markersize=3)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Metrics Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        ax3 = axes[1, 0]
        if 'lr/pg0' in df.columns:
            ax3.plot(df['epoch'], df['lr/pg0'], label='LR pg0', linewidth=2, color='orange')
        if 'lr/pg1' in df.columns:
            ax3.plot(df['epoch'], df['lr/pg1'], label='LR pg1', linewidth=2, color='red')
        if 'lr/pg2' in df.columns:
            ax3.plot(df['epoch'], df['lr/pg2'], label='LR pg2', linewidth=2, color='purple')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        final_metrics = {}
        if 'metrics/precision(B)' in df.columns:
            final_metrics['Precision'] = df['metrics/precision(B)'].iloc[-1]
        if 'metrics/recall(B)' in df.columns:
            final_metrics['Recall'] = df['metrics/recall(B)'].iloc[-1]
        if 'metrics/mAP50(B)' in df.columns:
            final_metrics['mAP50'] = df['metrics/mAP50(B)'].iloc[-1]
        if 'metrics/mAP50-95(B)' in df.columns:
            final_metrics['mAP50-95'] = df['metrics/mAP50-95(B)'].iloc[-1]
        
        if final_metrics:
            bars = ax4.bar(final_metrics.keys(), final_metrics.values(), 
                          color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
            ax4.set_ylabel('Score', fontsize=12)
            ax4.set_title('Final Model Performance', fontsize=14, fontweight='bold')
            ax4.set_ylim([0, 1])
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the custom plot
        custom_plot_path = results_dir / 'custom_training_summary.png'
        plt.savefig(custom_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Custom training summary saved to: {custom_plot_path}")
        plt.show()
        
        print("\n" + "="*50)
        print("FINAL TRAINING STATISTICS")
        print("="*50)
        print(f"Total Epochs Trained: {len(df)}")
        if final_metrics:
            for metric, value in final_metrics.items():
                print(f"{metric}: {value:.4f}")
        print("="*50 + "\n")
    
    else:
        print(f"Warning: Results CSV not found at {csv_path}")
    
    confusion_matrix_path = results_dir / 'confusion_matrix.png'
    if confusion_matrix_path.exists():
        print(f"✓ Confusion matrix saved to: {confusion_matrix_path}")
        
        # Display confusion matrix
        img = plt.imread(confusion_matrix_path)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Warning: Confusion matrix not found at {confusion_matrix_path}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print(f"All results saved in: {results_dir}")
    print("="*50)