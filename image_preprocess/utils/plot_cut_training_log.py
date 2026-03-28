import os, sys, re, argparse, csv
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    log_file = os.path.join(args.log_dir, 'training_log.txt')
    if not os.path.exists(log_file): sys.exit(f"Error: {log_file} no existe.")

    data = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    total_time = 0

    # egex to get the data from the log file
    log_re = re.compile(r'epoch:\s*(\d+).*?G_GAN:\s*([\d.]+)\s*D_real:\s*([\d.]+)\s*D_fake:\s*([\d.]+)\s*G:\s*([\d.]+)\s*NCE:\s*([\d.]+)(?:\s*NCE_Y:\s*([\d.]+))?')
    time_re = re.compile(r'Time Taken:\s*(\d+)\s*sec')

    with open(log_file, 'r') as f:
        for line in f:
            m = log_re.search(line)
            if m:
                ep = int(m.group(1))
                counts[ep] += 1
                for i, key in enumerate(['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'NCE_Y'], 2):
                    if m.group(i): data[ep][key] += float(m.group(i))
            
            t = time_re.search(line)
            if t: total_time += int(t.group(1))

    epochs = sorted(data.keys())
    metrics = {k: [data[e][k]/counts[e] for e in epochs] for k in ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'NCE_Y']}

    # Saving csv file
    stats_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../stats'))
    os.makedirs(stats_dir, exist_ok=True)
    csv_path = os.path.join(stats_dir, "cut_training_summary_log.csv")
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model_Name', 'Epoch'] + list(metrics.keys()))
        if f.tell() == 0: writer.writeheader()
        for i, ep in enumerate(epochs):
            row = {'Model_Name': args.model_name, 'Epoch': ep}
            row.update({k: round(metrics[k][i], 5) for k in metrics})
            writer.writerow(row)

    # Saving plot
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.set_title('Generator & NCE Losses')
    for k in ['G', 'G_GAN', 'NCE', 'NCE_Y']:
        if any(metrics[k]):
            ax1.plot(epochs, metrics[k], label=k, marker='.')
    ax1.legend(); ax1.set_ylabel('Loss')

    ax2.set_title('Discriminator Losses')
    for k in ['D_real', 'D_fake']:
        ax2.plot(epochs, metrics[k], label=k, marker='.')
    ax2.legend(); ax2.set_ylabel('Loss'); ax2.set_xlabel('Epoch')
    
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    fig.suptitle(f"Entrenamiento: {hours}h {minutes}m")
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, 'training_log.png'))

if __name__ == '__main__':
    main()